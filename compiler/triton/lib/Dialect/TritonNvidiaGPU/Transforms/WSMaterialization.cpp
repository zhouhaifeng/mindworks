/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "mlir/IR/OperationSupport.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"

#include <set>

using namespace mlir;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

enum class LoadType {
  Uninitialized,
  InsertSliceAsyncOp,
  InsertSliceAsyncV2Op,
  MultiKinds,
};

// This helper function returns the real threadId while ttng::GetThreadIdOp is
// actually threadId % 128 when warp specialization is enabled
Value getThreadId(OpBuilder &builder, Location loc) {
  Value threadId = builder.create<::mlir::gpu::ThreadIdOp>(
      loc, builder.getIndexType(), ::mlir::gpu::Dimension::x);
  auto cast = builder.create<UnrealizedConversionCastOp>(
      loc, TypeRange{builder.getIntegerType(32)}, ValueRange{threadId});
  return cast.getResult(0);
}

//===----------------------------------------------------------------------===//
// Materialize GetAgentIdOp
//===----------------------------------------------------------------------===//

void materializeGetAgentIdOp(Operation *parentOp) {
  parentOp->walk([](ttng::GetAgentIdOp op) {
    // In Hopper, each agent is a warpgroup consisting with 4 warps.
    auto loc = op.getLoc();
    OpBuilder builder(op);

    Value _128 = builder.create<arith::ConstantIntOp>(loc, 128, 32);
    Value threadId = getThreadId(builder, loc);
    Value agentId = builder.create<arith::DivUIOp>(loc, threadId, _128);
    op.getResult().replaceAllUsesWith(agentId);
    op->erase();

    // Update agent condition and insert "agent.num-warps"
    auto agentIdOp = agentId.getDefiningOp();
    builder.setInsertionPoint(agentIdOp);
    Value globalRoleId = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    int globalNumWarps = 0;
    for (auto cmpOp : agentIdOp->getUsers()) {
      assert(isa<arith::CmpIOp>(cmpOp));
      for (auto u : cmpOp->getUsers()) {
        if (isa<scf::IfOp>(u) && isa<triton::FuncOp>(u->getParentOp()) &&
            u->hasAttr("async_agent") && getAgentIds(u).size() == 1) {
          loc = u->getLoc();
          builder.setInsertionPoint(u);
          int numRoles = 1;
          if (u->hasAttr("agent.num-roles")) {
            numRoles =
                u->getAttrOfType<IntegerAttr>("agent.num-roles").getInt();
            // TODO: more flexible ways to get numWarps.
            auto numWarps = builder.getI32IntegerAttr(4 * numRoles);
            auto numWarpsBase = builder.getI32IntegerAttr(globalNumWarps);
            u->setAttr("agent.num-warps", numWarps);
            u->walk([&](ttng::GetMutexRoleIdOp roleIdOp) {
              roleIdOp->setAttr("agent.num-warps", numWarps);
              roleIdOp->setAttr("agent.num-warps-base", numWarpsBase);
            });
          }
          globalNumWarps += numRoles * 4;
          Value offset =
              builder.create<arith::ConstantIntOp>(loc, numRoles, 32);
          Value lowerBound = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::uge, agentId, globalRoleId);
          globalRoleId =
              builder.create<arith::AddIOp>(loc, globalRoleId, offset);
          Value upperBound = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::ult, agentId, globalRoleId);
          Value cond =
              builder.create<arith::AndIOp>(loc, lowerBound, upperBound);
          cmpOp->getResult(0).replaceAllUsesWith(cond);
          cmpOp->erase();
          break;
        }
      }
    }
  });
}

//===----------------------------------------------------------------------===//
// Materialize token operations
//===----------------------------------------------------------------------===//

LoadType scanLoadTypes(ttng::CreateTokenOp createTokenOp) {
  // TODO: Attach information of binded tensors to CreateTokenOp
  std::set<LoadType> loadTypes;
  createTokenOp->getBlock()->walk([&](Operation *op) {
    if (auto insertOp = dyn_cast<ttg::InsertSliceOp>(op)) {
      if (triton::isTensorPointerType(insertOp.getSrc().getType()))
        loadTypes.insert(LoadType::InsertSliceAsyncV2Op);
      else
        loadTypes.insert(LoadType::InsertSliceAsyncOp);
    } else if (isa<ttg::InsertSliceAsyncOp>(op)) {
      loadTypes.insert(LoadType::InsertSliceAsyncOp);
    } else if (isa<ttng::InsertSliceAsyncV2Op>(op)) {
      loadTypes.insert(LoadType::InsertSliceAsyncV2Op);
    }
  });
  assert(loadTypes.size() > 0 && "InsertSliceOp not found");
  assert(loadTypes.size() == 1 &&
         "Multiple kinds of load types are not supported");
  return *loadTypes.begin();
}

Value getMBarrierPhaseBit(OpBuilder &builder, Operation *op,
                          bool skipFirstWait) {
  // TODO: currently we only support one loop, no nested loop, while or
  // condition.
  auto loc = op->getLoc();
  auto forOp = op->getParentOfType<scf::ForOp>();
  if (!forOp) {
    return builder.create<arith::ConstantIntOp>(loc, skipFirstWait, 1);
  }

  auto defOp = op->getOperand(0).getDefiningOp();
  assert(isa<ttng::CreateTokenOp>(defOp) &&
         "mbarrier's definingOp is not createTokenOp");
  ttng::CreateTokenOp createTokenOp = dyn_cast<ttng::CreateTokenOp>(defOp);
  Value numStage =
      builder.create<arith::ConstantIntOp>(loc, createTokenOp.getNum(), 32);
  Value curStep = forOp.getBody()->getArguments().back();
  if (curStep.getType() == builder.getIndexType()) {
    curStep =
        builder.create<arith::IndexCastOp>(loc, numStage.getType(), curStep);
  }
  Value curPhase = builder.create<arith::DivUIOp>(loc, curStep, numStage);
  if (skipFirstWait) {
    // If skipFirstWait, it waits for phaseBit 1
    Value _1 = builder.create<arith::ConstantIntOp>(loc, 1, 32);
    curPhase = builder.create<arith::AddIOp>(loc, curPhase, _1);
  }
  Value _2 = builder.create<arith::ConstantIntOp>(loc, 2, 32);
  // TODO: May use alternative methods of phaseBit calculation to avoid high
  // overhead of RemOp
  Value phaseBit = builder.create<arith::RemUIOp>(loc, curPhase, _2);
  Value _0 = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, phaseBit,
                                       _0);
}

int getTxBytes(ttng::InsertSliceAsyncV2Op load) {
  // Both support ptr of tensor and tensor of ptr.
  RankedTensorType srcTensorType;
  if (auto srcType = dyn_cast<RankedTensorType>(load.getSrc().getType())) {
    srcTensorType = srcType;
  } else if (auto srcType =
                 dyn_cast<triton::PointerType>(load.getSrc().getType())) {
    srcTensorType = dyn_cast<RankedTensorType>(srcType.getPointeeType());
  } else {
    llvm_unreachable("Unexpected src type");
  }
  auto shapePerCTA = ttg::getShapePerCTA(srcTensorType);
  auto elemTy =
      dyn_cast<RankedTensorType>(load.getDst().getType()).getElementType();
  int bytesPerElem = elemTy.getIntOrFloatBitWidth() / 8;
  return product<int64_t>(shapePerCTA) * bytesPerElem;
}

int applyCommit(OpBuilder &builder, ttng::ProducerCommitOp &op,
                Value mbarrier) {
  // TODO: currently it only handles loads in ProducerCommitOp's nearest parent
  // block. Neither support multiple ProducerCommitOp, e.g. fused attention,
  // epilogue fusion.
  int txCnt = 0;
  SmallVector<Operation *> deprecatedOps;
  auto agentIds = getAgentIds(op);
  // Materialize InsertSliceOp
  for (auto &ItrOp : op->getBlock()->getOperations()) {
    // Check operations before ProducerCommitOp
    if (OperationEquivalence::isEquivalentTo(&ItrOp, op.getOperation(),
                                             OperationEquivalence::None)) {
      break;
    }
    if (auto insertOp = dyn_cast<ttg::InsertSliceOp>(ItrOp)) {
      deprecatedOps.push_back(&ItrOp);
      builder.setInsertionPoint(insertOp);
      if (!::mlir::triton::isTensorPointerType(insertOp.getSrc().getType())) {
        // Transform to InsertSliceAsyncOp
        auto newSliceOp = builder.create<triton::gpu::InsertSliceAsyncOp>(
            /*loc=*/insertOp.getLoc(), /*result=*/insertOp.getDst().getType(),
            /*src=*/insertOp.getSrc(), /*dst=*/insertOp.getDst(),
            /*index=*/insertOp.getIndex(),
            /*mask=*/insertOp.getMask(), /*other=*/insertOp.getOther(),
            /*cache=*/insertOp.getCache(), /*evict=*/insertOp.getEvict(),
            /*isVolatile=*/insertOp.getIsVolatile(),
            /*axis=*/insertOp.getAxis());
        insertOp.getResult().replaceAllUsesWith(newSliceOp.getResult());
        setAgentIds(newSliceOp, agentIds);
      } else {
        // Transform to InsertSliceAsyncV2Op
        auto extractBarrierOp = dyn_cast<ttng::ExtractMBarrierOp>(
            builder.clone(*(mbarrier.getDefiningOp())));
        auto newSliceOp = builder.create<ttng::InsertSliceAsyncV2Op>(
            /*loc=*/insertOp.getLoc(), /*result=*/insertOp.getDst().getType(),
            /*src=*/insertOp.getSrc(), /*dst=*/insertOp.getDst(),
            /*index=*/insertOp.getIndex(),
            /*mbar*/ extractBarrierOp.getResult(), /*mask=*/insertOp.getMask(),
            /*other=*/insertOp.getOther(),
            /*cache=*/insertOp.getCache(), /*evict=*/insertOp.getEvict(),
            /*isVolatile=*/insertOp.getIsVolatile(),
            /*axis=*/insertOp.getAxis());
        insertOp.getResult().replaceAllUsesWith(newSliceOp.getResult());
        setAgentIds(newSliceOp, agentIds);
        txCnt += getTxBytes(newSliceOp);
      }
    }
  }
  builder.setInsertionPoint(op);
  for (auto d : deprecatedOps) {
    d->erase();
  }

  return txCnt;
}

void processProducerAcquireOp(OpBuilder &builder, ttng::ProducerAcquireOp op,
                              Value bufferEmpty) {
  auto loc = op.getLoc();
  // The first producer_aquire should be met immediately, so initailly producer
  // skips the fisrt wait
  Value phase = getMBarrierPhaseBit(builder, op, 1);
  auto waitOp = builder.create<ttng::MBarrierWaitOp>(loc, bufferEmpty, phase);
  assert(op.getOperation()->hasAttr("async_agent"));
  setAgentIds(waitOp, getAgentIds(op.getOperation()));
}

void processProducerCommitOp(OpBuilder &builder, ttng::ProducerCommitOp op,
                             Value bufferFull, LoadType loadType) {
  auto loc = op.getLoc();
  int txCnt = applyCommit(builder, op, bufferFull);
  ttng::MBarrierArriveOp arriveOp;

  if (loadType == LoadType::InsertSliceAsyncOp) {
    // Each thread arrives
    Value pred = builder.create<arith::ConstantIntOp>(loc, 1, 1);
    arriveOp = builder.create<ttng::MBarrierArriveOp>(
        loc, bufferFull, pred, /*remoteCTAId*/ nullptr, /*trackAsyncOp*/ true,
        txCnt);
  } else {
    // Only thread 0 arrives
    Value _0 = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    Value threadId = getThreadId(builder, loc);
    Value pred = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               threadId, _0);
    arriveOp = builder.create<ttng::MBarrierArriveOp>(
        loc, bufferFull, pred, /*remoteCTAId*/ nullptr, /*trackAsyncOp*/ false,
        txCnt);
  }

  assert(op.getOperation()->hasAttr("async_agent"));
  setAgentIds(arriveOp, getAgentIds(op.getOperation()));
}

void processConsumerWaitOp(OpBuilder &builder, ttng::ConsumerWaitOp op,
                           Value bufferFull) {
  auto loc = op.getLoc();
  Value phase = getMBarrierPhaseBit(builder, op, 0);
  auto waitOp = builder.create<ttng::MBarrierWaitOp>(loc, bufferFull, phase);
  assert(op.getOperation()->hasAttr("async_agent"));
  setAgentIds(waitOp, getAgentIds(op.getOperation()));
}

void processConsumerReleaseOp(OpBuilder &builder, ttng::ConsumerReleaseOp op,
                              Value bufferEmpty, int numCTAs) {
  auto loc = op.getLoc();

  // Constants
  Value _0 = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value _4 = builder.create<arith::ConstantIntOp>(loc, 4, 32);
  Value _8 = builder.create<arith::ConstantIntOp>(loc, 8, 32);
  Value _32 = builder.create<arith::ConstantIntOp>(loc, 32, 32);
  Value _128 = builder.create<arith::ConstantIntOp>(loc, 128, 32);

  // threadId = threadId % 128
  Value threadId =
      builder.create<arith::RemUIOp>(loc, getThreadId(builder, loc), _128);

  // k = threadId / 8
  Value k = builder.create<arith::DivUIOp>(loc, threadId, _8);

  // row = k / 4
  Value row = builder.create<arith::DivUIOp>(loc, k, _4);

  // col = k % 4
  Value col = builder.create<arith::RemUIOp>(loc, k, _4);

  // remoteCTAId = (col ^ row) * 4 + col
  Value remoteCTAId = builder.create<arith::AddIOp>(
      loc,
      Value{builder.create<arith::MulIOp>(
          loc, Value{builder.create<arith::XOrIOp>(loc, col, row)}, _4)},
      col);

  // pred0 = threadId % 8 == 0
  Value pred0 = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq,
      builder.create<arith::RemUIOp>(loc, threadId, _8), _0);

  // pred1 = remoteCTAId < numCTAs
  Value pred1 = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ult, remoteCTAId,
      builder.create<arith::ConstantIntOp>(loc, numCTAs, 32));

  // pred = pred0 & pred1
  Value pred = builder.create<arith::AndIOp>(loc, pred0, pred1);

  // bufferEmpty arrive
  auto arriveOp = builder.create<ttng::MBarrierArriveOp>(loc, bufferEmpty, pred,
                                                         remoteCTAId, false, 0);

  assert(op.getOperation()->hasAttr("async_agent"));
  setAgentIds(arriveOp, getAgentIds(op.getOperation()));
}

void materializeTokenOperations(Operation *parentOp, int numCTAs) {
  SmallVector<Operation *> deprecatedOps;
  parentOp->walk([&](ttng::CreateTokenOp createTokenOp) {
    // Scan load type
    LoadType loadType = scanLoadTypes(createTokenOp);

    // mBarrierTy
    MLIRContext *context = createTokenOp.getContext();
    auto i64Ty = IntegerType::get(context, 64);
    auto mBarrierTy = triton::PointerType::get(i64Ty, 3);

    // mBarriersTy
    auto CTALayout = ttg::CTALayoutAttr::get(context, {1}, {1}, {0});
    auto sharedLayout =
        ttg::SharedEncodingAttr::get(context, 1, 1, 1, {0}, CTALayout, false);
    auto mBarriersTy =
        RankedTensorType::get({createTokenOp.getNum()}, i64Ty, sharedLayout);

    // Process CreateTokenOp
    OpBuilder builder(createTokenOp);
    auto tokenLoc = createTokenOp.getLoc();
    unsigned bufferFullCount =
        loadType == LoadType::InsertSliceAsyncV2Op ? 1 : 128;
    Value bufferFullArray = builder.create<ttng::AllocMBarrierOp>(
        tokenLoc, mBarriersTy, bufferFullCount);
    Value bufferEmptyArray =
        builder.create<ttng::AllocMBarrierOp>(tokenLoc, mBarriersTy, numCTAs);

    if (numCTAs == 1) {
      builder.create<mlir::gpu::BarrierOp>(tokenLoc);
    } else {
      // Make sure that MBarriers are initialized in all CTAs
      builder.create<triton::nvidia_gpu::ClusterArriveOp>(tokenLoc, false);
      builder.create<triton::nvidia_gpu::ClusterWaitOp>(tokenLoc);
    }

    // Helper function for extracting bufferFull
    auto extractBufferFull = [&](Location loc, Value idx) -> Value {
      return builder.create<ttng::ExtractMBarrierOp>(loc, mBarrierTy,
                                                     bufferFullArray, idx);
    };

    // Helper function for extracting bufferEmpty
    auto extractBufferEmpty = [&](Location loc, Value idx) -> Value {
      return builder.create<ttng::ExtractMBarrierOp>(loc, mBarrierTy,
                                                     bufferEmptyArray, idx);
    };

    // Process token users
    for (Operation *user : createTokenOp.getResult().getUsers()) {
      auto loc = user->getLoc();
      builder.setInsertionPoint(user);
      if (auto op = dyn_cast<ttng::ProducerAcquireOp>(user)) {
        Value bufferEmpty = extractBufferEmpty(loc, op.getIdx());
        assert(user->hasAttr("async_agent"));
        setAgentIds(bufferEmpty.getDefiningOp(), getAgentIds(user));
        processProducerAcquireOp(builder, op, bufferEmpty);
      } else if (auto op = dyn_cast<ttng::ProducerCommitOp>(user)) {
        Value bufferFull = extractBufferFull(loc, op.getIdx());
        assert(user->hasAttr("async_agent"));
        setAgentIds(bufferFull.getDefiningOp(), getAgentIds(user));
        processProducerCommitOp(builder, op, bufferFull, loadType);
      } else if (auto op = dyn_cast<ttng::ConsumerWaitOp>(user)) {
        Value bufferFull = extractBufferFull(loc, op.getIdx());
        assert(user->hasAttr("async_agent"));
        setAgentIds(bufferFull.getDefiningOp(), getAgentIds(user));
        processConsumerWaitOp(builder, op, bufferFull);
      } else if (auto op = dyn_cast<ttng::ConsumerReleaseOp>(user)) {
        Value bufferEmpty = extractBufferEmpty(loc, op.getIdx());
        assert(user->hasAttr("async_agent"));
        setAgentIds(bufferEmpty.getDefiningOp(), getAgentIds(user));
        processConsumerReleaseOp(builder, op, bufferEmpty, numCTAs);
      } else {
        llvm_unreachable("Unexpected user of token");
      }
      deprecatedOps.push_back(user);
    }

    deprecatedOps.push_back(createTokenOp);
  });
  for (auto op : deprecatedOps) {
    op->erase();
  }

  // Insert a cluster barrier before the kernel exits. Without this barrier,
  // mbarrier_remote_arrive will fail if the remote CTA already exits.
  if (numCTAs > 1) {
    parentOp->walk([&](triton::FuncOp funcOp) {
      Block *block = &funcOp.getBody().front();
      auto returnOp = llvm::cast<triton::ReturnOp>(block->getTerminator());
      OpBuilder builder(returnOp);
      auto loc = returnOp.getLoc();
      builder.create<triton::nvidia_gpu::ClusterArriveOp>(loc, false);
      builder.create<triton::nvidia_gpu::ClusterWaitOp>(loc);
    });
  }
}

//===----------------------------------------------------------------------===//
// Materialize mutex operations
//===----------------------------------------------------------------------===//

void mutexSyncPingPang(Operation *parentOp, int numAgents, int &nameBarrierId,
                       int &globalNumRoles) {
  // ping-pang mutex sync: using named barrier and only suitable for two roles.
  // Take mutex syncronization between dot and store as an example:
  // * For dot loop:
  //   * role 0 waits for named barrier 15 (loop enter), arrives named barrier
  //   14 (loop leave)
  //   * role 1 waits for named barrier 14 (loop enter), arrives named barrier
  //   15 (loop leave)
  // * For store:
  //   * role 0 waits for named barrier 13 (store enter), arrives named barrier
  //   12 (store leave)
  //   * role 1 waits for named barrier 12 (store enter), arrives named barrier
  //   13 (store leave)
  // As number of named barriers is limited (16), theoretically this mechanism
  // only support few roles and agents.
  int numRoles = 2, times = 0;
  globalNumRoles += numRoles;
  Value roleId;
  parentOp->walk([&](ttng::GetMutexRoleIdOp getMutexRoleIdOp) {
    // GetMutexRoleIdOp only occures once.
    assert(times == 0);
    OpBuilder builder(getMutexRoleIdOp);
    numRoles = getMutexRoleIdOp.getNum();
    auto loc = getMutexRoleIdOp->getLoc();
    Value threadId = getThreadId(builder, loc);
    assert(getMutexRoleIdOp->hasAttr("agent.num-warps"));
    int numThreads =
        32 * getMutexRoleIdOp->getAttrOfType<IntegerAttr>("agent.num-warps")
                 .getInt();
    int numThreadsBase =
        32 *
        getMutexRoleIdOp->getAttrOfType<IntegerAttr>("agent.num-warps-base")
            .getInt();
    assert(numThreads % numRoles == 0);
    // TODO: more flexible ways to determine numWarps of each agent.
    Value numThreadsValue =
        builder.create<arith::ConstantIntOp>(loc, numThreads, 32);
    Value numRolesValue =
        builder.create<arith::ConstantIntOp>(loc, numRoles, 32);
    Value numThreadsBaseValue =
        builder.create<arith::ConstantIntOp>(loc, numThreadsBase, 32);
    Value numThreadsPerRole =
        builder.create<arith::DivUIOp>(loc, numThreadsValue, numRolesValue);
    Value numRemThreads =
        builder.create<arith::SubIOp>(loc, threadId, numThreadsBaseValue);
    roleId =
        builder.create<arith::DivUIOp>(loc, numRemThreads, numThreadsPerRole);
    getMutexRoleIdOp.getResult().replaceAllUsesWith(roleId);
    getMutexRoleIdOp->erase();
    times++;
  });

  parentOp->walk<WalkOrder::PreOrder>([&](ttng::CreateMutexOp createMutexOp) {
    // Currently, inner-agent sync counts from barId 1 (see membar.cpp, bar 0
    // is used for whole block sync).
    // We need to guarantee mutex sync won't use bars of inner-agent sync.
    assert(nameBarrierId > globalNumRoles);
    // Process CreateMutexOp
    OpBuilder builder(createMutexOp);
    // TODO: change the hard code of numThreads
    auto loc = createMutexOp->getLoc();
    Value numThreads = builder.create<arith::ConstantIntOp>(loc, 256, 32);
    Value _0 = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    Value isRole0 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  roleId, _0);
    assert(nameBarrierId < nameBarrierIdEnd &&
           nameBarrierId - 1 >= nameBarrierIdBegin);
    Value namedBarrierId0 =
        builder.create<arith::ConstantIntOp>(loc, nameBarrierId, 32);
    Value namedBarrierId1 =
        builder.create<arith::ConstantIntOp>(loc, nameBarrierId - 1, 32);
    // Process mutex users
    int numUsers = 0;
    for (Operation *user : createMutexOp.getResult().getUsers()) {
      numUsers++;
      assert(numUsers <= 2);
      auto loc = user->getLoc();
      builder.setInsertionPoint(user);
      if (auto op = dyn_cast<ttng::LockOp>(user)) {
        Value barEnter = builder.create<arith::SelectOp>(
            loc, isRole0, namedBarrierId0, namedBarrierId1);
        builder.create<ttng::NamedBarrierWaitOp>(loc, barEnter, numThreads);
      } else if (auto op = dyn_cast<ttng::UnlockOp>(user)) {
        Value barLeave = builder.create<arith::SelectOp>(
            loc, isRole0, namedBarrierId1, namedBarrierId0);
        builder.create<ttng::NamedBarrierArriveOp>(loc, barLeave, numThreads);
      } else
        llvm_unreachable("Unexpected user of mutex");
      user->erase();
    }
    nameBarrierId -= 2;
    nameBarrierIdEnd -= 2;
    createMutexOp.erase();
  });
}

void processLockOp(OpBuilder &builder, ttng::LockOp op) {
  auto loc = op.getLoc();
  assert(op->hasAttr("mutex.barId") && op->hasAttr("mutex.numThreads"));
  auto barIds = getMutexBarIds(op);
  auto threads = getMutexNumThreads(op);
  assert(barIds.size() > 0 && barIds.size() == threads.size());
  for (int i = 0; i < barIds.size(); ++i) {
    Value numThreads =
        builder.create<arith::ConstantIntOp>(loc, threads[i], 32);
    Value barrier = builder.create<arith::ConstantIntOp>(loc, barIds[i], 32);
    builder.create<ttng::NamedBarrierWaitOp>(loc, barrier, numThreads);
  }
}

void processUnlockOp(OpBuilder &builder, ttng::UnlockOp op) {
  auto loc = op.getLoc();
  assert(op->hasAttr("mutex.barId") && op->hasAttr("mutex.numThreads"));
  auto barIds = getMutexBarIds(op);
  auto threads = getMutexNumThreads(op);
  assert(barIds.size() > 0 && barIds.size() == threads.size());
  for (int i = 0; i < barIds.size(); ++i) {
    Value numThreads =
        builder.create<arith::ConstantIntOp>(loc, threads[i], 32);
    Value barrier = builder.create<arith::ConstantIntOp>(loc, barIds[i], 32);
    builder.create<ttng::NamedBarrierArriveOp>(loc, barrier, numThreads);
  }
}

void materializeMutexOperationsOthers(ModuleOp parentOp) {
  parentOp->walk([](ttng::CreateMutexOp createMutexOp) {
    // Process CreateMutexOp
    OpBuilder builder(createMutexOp);

    // Process mutex users
    for (Operation *user : createMutexOp.getResult().getUsers()) {
      auto loc = user->getLoc();
      builder.setInsertionPoint(user);
      if (auto op = dyn_cast<ttng::LockOp>(user))
        processLockOp(builder, op);
      else if (auto op = dyn_cast<ttng::UnlockOp>(user))
        processUnlockOp(builder, op);
      else
        llvm_unreachable("Unexpected user of mutex");
      user->erase();
    }

    createMutexOp.erase();
  });
}

void materializeMutexOperations(ModuleOp parentOp) {
  nameBarrierIdEnd = 16;
  int nameBarrierId = 15;
  int globalNumRoles = 0;
  // Materialize mutex operations from WSMutex, i.e. auto-mutex
  parentOp->walk([&](scf::IfOp IfOp) {
    int numRoles = 0;
    if (IfOp->hasAttr("agent.num-roles")) {
      assert(parentOp->hasAttr("async.num-agents"));
      int numAgents =
          parentOp->getAttrOfType<IntegerAttr>("async.num-agents").getInt();
      numRoles = IfOp->getAttrOfType<IntegerAttr>("agent.num-roles").getInt();
      // TODO: To support arbitrary number of roles, use mbarrier.
      assert(numRoles == 2);
      mutexSyncPingPang(IfOp, numAgents, nameBarrierId, globalNumRoles);
    }
  });
  // Materialize mutex operations for remaining cases.
  // User needs to guarantee correctness of synchronization.
  materializeMutexOperationsOthers(parentOp);
}

// TODO: may also not support 8-warp kernel.
void tryRegisterRealloc(ModuleOp mod) {
  constexpr int LoadRegisterRequirement = 40;
  constexpr int MmaRegisterRequirement = 232;
  OpBuilderWithAgentIds builder(mod.getContext());

  auto isLoadAgent = [](scf::IfOp ifOp) -> bool {
    return ifOp
        ->walk([](Operation *op) {
          if (isa<ttg::InsertSliceOp, ttg::InsertSliceAsyncOp,
                  ttng::InsertSliceAsyncV2Op>(op))
            return WalkResult::interrupt();
          return WalkResult::advance();
        })
        .wasInterrupted();
  };

  auto isMmaAgent = [](scf::IfOp ifOp) -> bool {
    return ifOp
        ->walk([](Operation *op) {
          if (isa<triton::DotOp, ttng::DotAsyncOp>(op))
            return WalkResult::interrupt();
          return WalkResult::advance();
        })
        .wasInterrupted();
  };

  // TODO: we need to make agent info more handy
  SmallVector<scf::IfOp> agentOps;
  mod->walk([&agentOps](triton::FuncOp funcOp) {
    Block *block = &funcOp.getBody().front();
    for (Operation &op : block->getOperations()) {
      if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
        if (getAgentIds(ifOp).size() == 1) {
          agentOps.push_back(ifOp);
        }
      }
    }
  });
  for (auto ifOp : agentOps) {
    builder.setInsertionPointToStart(&(ifOp.getThenRegion().front()));
    builder.setAgentIdsFromOp(ifOp);
    auto loc = ifOp.getLoc();
    Type i32_ty = builder.getIntegerType(32);
    // If an agent has both mma and load, do nothing.
    if (isMmaAgent(ifOp) && isLoadAgent(ifOp))
      continue;
    if (isMmaAgent(ifOp)) {
      builder.createWithAgentIds<ttng::RegAllocOp>(
          loc, builder.getIntegerAttr(i32_ty, MmaRegisterRequirement));
    } else if (isLoadAgent(ifOp)) {
      builder.createWithAgentIds<ttng::RegDeallocOp>(
          loc, builder.getIntegerAttr(i32_ty, LoadRegisterRequirement));
    }
  }
}

//===----------------------------------------------------------------------===//
// WSMaterializationPass
//===----------------------------------------------------------------------===//

struct WSMaterializationPass
    : public TritonGPUWSMaterializationBase<WSMaterializationPass> {
  WSMaterializationPass() = default;
  WSMaterializationPass(int computeCapability) {
    this->computeCapability = computeCapability;
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!ttng::TritonNvidiaGPUDialect::getWSSupportedAttr(mod))
      return signalPassFailure();

    if (computeCapability / 10 < 9) {
      llvm_unreachable("WSMaterialization pass only supports sm_9x as of now.");
      signalPassFailure();
    }

    int numCTAs = ttg::TritonGPUDialect::getNumCTAs(mod);

    materializeGetAgentIdOp(mod);
    materializeTokenOperations(mod, numCTAs);
    materializeMutexOperations(mod);
    tryRegisterRealloc(mod);

    // TODO: More flexible way to set num-warps
    // One dma, one math warp group, set num-warps = 8
    auto i32_ty = IntegerType::get(mod->getContext(), 32);
    mod->setAttr("triton_gpu.num-warps",
                 IntegerAttr::get(i32_ty, llvm::APInt(32, 8)));

    WalkResult result = mod->walk([&](scf::IfOp ifOp) {
      if (ifOp->hasAttr("agent.num-roles")) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      mod->setAttr("triton_gpu.num-warps",
                   IntegerAttr::get(i32_ty, llvm::APInt(32, 12)));
    }
    mod->removeAttr("async.num-agents");
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// createTritonNvidiaGPUWSMaterializationPass
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass>
mlir::createTritonNvidiaGPUWSMaterializationPass(int computeCapability) {
  return std::make_unique<WSMaterializationPass>(computeCapability);
}
