// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/multiclass_nms.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <vector>

#include "../shape_inference/include/multiclass_nms_shape_inference.hpp"
#include "ngraph/runtime/reference/multiclass_nms.hpp"
#include "ngraph/runtime/reference/utils/nms_common.hpp"
#include "ngraph/shape.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
namespace multiclass_nms_impl {

namespace {
std::vector<float> get_floats(const std::shared_ptr<HostTensor>& input, const Shape& shape) {
    size_t input_size = shape_size(shape);
    std::vector<float> result(input_size);

    switch (input->get_element_type()) {
    case element::Type_t::bf16: {
        bfloat16* p = input->get_data_ptr<bfloat16>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = float(p[i]);
        }
    } break;
    case element::Type_t::f16: {
        float16* p = input->get_data_ptr<float16>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = float(p[i]);
        }
    } break;
    case element::Type_t::f32: {
        float* p = input->get_data_ptr<float>();
        memcpy(result.data(), p, input_size * sizeof(float));
    } break;
    default:
        throw std::runtime_error("Unsupported data type.");
        break;
    }

    return result;
}

std::vector<int64_t> get_integers(const std::shared_ptr<HostTensor>& input, const Shape& shape) {
    size_t input_size = shape_size(shape);
    std::vector<int64_t> result(input_size);

    switch (input->get_element_type()) {
    case element::Type_t::i8: {
        auto p = input->get_data_ptr<int8_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::i16: {
        auto p = input->get_data_ptr<int16_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::i32: {
        auto p = input->get_data_ptr<int32_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::i64: {
        auto p = input->get_data_ptr<int64_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::u8: {
        auto p = input->get_data_ptr<uint8_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::u16: {
        auto p = input->get_data_ptr<uint16_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::u32: {
        auto p = input->get_data_ptr<uint32_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::u64: {
        auto p = input->get_data_ptr<uint64_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    default:
        throw std::runtime_error("Unsupported data type");
        break;
    }

    return result;
}

static std::vector<float> prepare_boxes_data(const std::shared_ptr<HostTensor>& boxes, const Shape& boxes_shape) {
    auto result = get_floats(boxes, boxes_shape);
    return result;
}

static std::vector<float> prepare_scores_data(const std::shared_ptr<HostTensor>& scores, const Shape& scores_shape) {
    auto result = get_floats(scores, scores_shape);
    return result;
}

static std::vector<int64_t> prepare_roisnum_data(const std::shared_ptr<HostTensor>& roisnum,
                                                 const Shape& roisnum_shape) {
    auto result = get_integers(roisnum, roisnum_shape);
    return result;
}

}  // namespace

constexpr size_t boxes_port = 0;
constexpr size_t scores_port = 1;
constexpr size_t roisnum_port = 2;

InfoForNMS get_info_for_nms_eval(const std::shared_ptr<op::util::MulticlassNmsBase>& nms,
                                 const std::vector<std::shared_ptr<HostTensor>>& inputs) {
    InfoForNMS result;

    const auto boxes_ps = inputs[boxes_port]->get_partial_shape();
    const auto scores_ps = inputs[scores_port]->get_partial_shape();
    std::vector<PartialShape> input_shapes = {boxes_ps, scores_ps};
    if (nms->get_input_size() == 3) {
        const auto roisnum_ps = inputs[roisnum_port]->get_partial_shape();
        input_shapes.push_back(roisnum_ps);
    }

    std::vector<PartialShape> output_shapes = {{Dimension::dynamic(), 6},
                                               {Dimension::dynamic(), 1},
                                               {Dimension::dynamic()}};
    ov::op::util::shape_infer(nms.get(),
                              input_shapes,
                              output_shapes,
                              true,
                              false);  // here just for upper boundary estimation.

    result.selected_outputs_shape = output_shapes[0].to_shape();
    result.selected_indices_shape = output_shapes[1].to_shape();
    result.selected_numrois_shape = output_shapes[2].to_shape();

    result.boxes_shape = inputs[boxes_port]->get_shape();
    result.scores_shape = inputs[scores_port]->get_shape();

    result.boxes_data = prepare_boxes_data(inputs[boxes_port], result.boxes_shape);
    result.scores_data = prepare_scores_data(inputs[scores_port], result.scores_shape);

    if (inputs.size() == 3) {
        result.roisnum_shape = inputs[roisnum_port]->get_shape();
        result.roisnum_data = prepare_roisnum_data(inputs[roisnum_port], result.roisnum_shape);
    }

    result.selected_outputs_shape_size = shape_size(result.selected_outputs_shape);
    result.selected_indices_shape_size = shape_size(result.selected_indices_shape);
    result.selected_numrois_shape_size = shape_size(result.selected_numrois_shape);

    return result;
}

using Rectangle = runtime::reference::nms_common::Rectangle;
using BoxInfo = runtime::reference::nms_common::BoxInfo;
static float intersectionOverUnion(const Rectangle& boxI, const Rectangle& boxJ, const bool normalized) {
    const float norm = static_cast<float>(normalized == false);

    float areaI = (boxI.y2 - boxI.y1 + norm) * (boxI.x2 - boxI.x1 + norm);
    float areaJ = (boxJ.y2 - boxJ.y1 + norm) * (boxJ.x2 - boxJ.x1 + norm);

    if (areaI <= 0.0f || areaJ <= 0.0f) {
        return 0.0f;
    }

    float intersection_ymin = std::max(boxI.y1, boxJ.y1);
    float intersection_xmin = std::max(boxI.x1, boxJ.x1);
    float intersection_ymax = std::min(boxI.y2, boxJ.y2);
    float intersection_xmax = std::min(boxI.x2, boxJ.x2);

    float intersection_area = std::max(intersection_ymax - intersection_ymin + norm, 0.0f) *
                              std::max(intersection_xmax - intersection_xmin + norm, 0.0f);

    return intersection_area / (areaI + areaJ - intersection_area);
}

// slice an image
// for the case when boxes are not shared among classes.
// boxes: [in] C, M, 4 -> [out] C, M', 4
// scores: [in] C, M -> [out] C, M'
// start: start index along axis "M"
template <class T>
std::vector<T> slice_image(const T* data, const Shape& data_shape, const int64_t start, const int64_t item_num) {
    std::vector<T> slice_data;
    const auto class_num = data_shape[0];
    const auto item_size = (data_shape.size() == 3) ? data_shape[2] : 1;

    // start and end should within [0, M)
    OPENVINO_ASSERT(start >= 0 && (start + item_num) <= static_cast<int64_t>(data_shape[1]),
                    "Invaid inputs as it is trying to slice data out of range.");

    const auto row_num = item_num * item_size;
    slice_data.reserve(class_num * row_num);
    T* item_data = slice_data.data();
    T* src = const_cast<T*>(data + start * item_size);
    for (size_t i = 0; i < class_num; i++) {
        std::memcpy(item_data + i * row_num, src, sizeof(T) * row_num);
        src += data_shape[1] * item_size;
    }

    return slice_data;
}

// nms over a class
// boxes:       num_priors, 4
// scores:      num_priors, 1
static const std::vector<BoxInfo> nms(const float* boxes_data,
                                      const float* scoresPtr,
                                      const int64_t num_priors,
                                      const op::util::MulticlassNmsBase::Attributes& attrs,
                                      const size_t image_idx,
                                      const size_t class_idx) {
    auto func = [](float iou, float adaptive_threshold) {
        return iou <= adaptive_threshold ? 1.0f : 0.0f;
    };

    const Rectangle* bboxesPtr = reinterpret_cast<const Rectangle*>(const_cast<float*>(boxes_data));
    std::vector<BoxInfo> selected;  // container for a class

    auto adaptive_threshold = attrs.iou_threshold;

    std::vector<BoxInfo> candidate_boxes;
    for (int64_t box_idx = 0; box_idx < num_priors; box_idx++) {
        if (scoresPtr[box_idx] >= attrs.score_threshold) { /* NOTE: ">=" instead of ">" used in PDPD */
            candidate_boxes.emplace_back(bboxesPtr[box_idx], box_idx, scoresPtr[box_idx], 0, image_idx, class_idx);
        }
    }

    int candiate_size = static_cast<int>(candidate_boxes.size());

    // threshold nms_top_k for each class
    // NOTE: "nms_top_k" in PDPD not exactly equal to
    // "max_output_boxes_per_class" in ONNX.
    if (attrs.nms_top_k > -1 && attrs.nms_top_k < candiate_size) {
        candiate_size = attrs.nms_top_k;
    }

    if (candiate_size <= 0) {  // early drop
        return selected;       // empty
    }

    // sort by score in current class
    std::partial_sort(candidate_boxes.begin(),
                      candidate_boxes.begin() + candiate_size,
                      candidate_boxes.end(),
                      std::greater<BoxInfo>());

    std::priority_queue<BoxInfo> sorted_boxes(candidate_boxes.begin(),
                                              candidate_boxes.begin() + candiate_size,
                                              std::less<BoxInfo>());

    // Get the next box with top score, filter by iou_threshold
    BoxInfo next_candidate;
    float original_score;

    while (!sorted_boxes.empty()) {
        next_candidate = sorted_boxes.top();
        original_score = next_candidate.score;
        sorted_boxes.pop();

        bool should_hard_suppress = false;
        for (int64_t j = static_cast<int64_t>(selected.size()) - 1; j >= next_candidate.suppress_begin_index; --j) {
            float iou = intersectionOverUnion(next_candidate.box, selected[j].box, attrs.normalized);
            next_candidate.score *= func(iou, adaptive_threshold);

            if (iou >= adaptive_threshold) {
                should_hard_suppress = true;
                break;
            }

            if (next_candidate.score <= attrs.score_threshold) {
                break;
            }
        }

        next_candidate.suppress_begin_index = selected.size();

        if (!should_hard_suppress) {
            if (attrs.nms_eta < 1 && adaptive_threshold > 0.5) {
                adaptive_threshold *= attrs.nms_eta;
            }
            if (next_candidate.score == original_score) {
                selected.push_back(next_candidate);
                continue;
            }
            if (next_candidate.score > attrs.score_threshold) {
                sorted_boxes.push(next_candidate);
            }
        }
    }

    return selected;
}

// nms over an image
// shared           Y               N
// boxes:       1, M, 4         C, M', 4
// scores:      1, C, M         C, M'
static const std::vector<BoxInfo> multiclass_nms(const float* boxes_data,
                                                 const Shape& boxes_data_shape,
                                                 const float* scores_data,
                                                 const Shape& scores_data_shape,
                                                 const op::util::MulticlassNmsBase::Attributes& attrs,
                                                 const size_t image_idx,
                                                 const bool shared) {
    int64_t num_dets = 0;
    std::vector<BoxInfo> selected_boxes;  // container for a batch element
    (void)boxes_data_shape;

    const auto num_classes = shared ? scores_data_shape[1] : scores_data_shape[0];
    const auto num_boxes = shared ? scores_data_shape[2] : scores_data_shape[1];

    for (size_t class_idx = 0; class_idx < num_classes; class_idx++) {
        if (static_cast<int>(class_idx) == attrs.background_class)
            continue;

        std::vector<BoxInfo> selected;
        if (shared) {
            const float* scoresPtr = scores_data + class_idx * num_boxes;
            selected = nms(boxes_data, scoresPtr, num_boxes, attrs, image_idx, class_idx);
        } else {
            const float* scoresPtr = scores_data + class_idx * num_boxes;
            const float* boxesPtr = boxes_data + class_idx * num_boxes * 4;
            selected = nms(boxesPtr, scoresPtr, num_boxes, attrs, image_idx, class_idx);
        }

        for (const auto& box_info : selected) {
            selected_boxes.push_back(box_info);
        }
        num_dets += selected.size();
    }  // for each class

    // sort inside batch element before go through keep_top_k
    std::sort(selected_boxes.begin(), selected_boxes.end(), [](const BoxInfo& l, const BoxInfo& r) {
        return ((l.batch_index == r.batch_index) &&
                ((l.score > r.score) || ((std::fabs(l.score - r.score) < 1e-6) && l.class_index < r.class_index) ||
                 ((std::fabs(l.score - r.score) < 1e-6) && l.class_index == r.class_index && l.index < r.index)));
    });

    // threshold keep_top_k for each batch element
    if (attrs.keep_top_k > -1 && attrs.keep_top_k < num_dets) {
        num_dets = attrs.keep_top_k;
        selected_boxes.resize(num_dets);
    }

    // sort
    if (!attrs.sort_result_across_batch) {
        if (attrs.sort_result_type == op::util::MulticlassNmsBase::SortResultType::CLASSID) {
            std::sort(selected_boxes.begin(), selected_boxes.end(), [](const BoxInfo& l, const BoxInfo& r) {
                return (
                    (l.batch_index == r.batch_index) &&
                    ((l.class_index < r.class_index) || ((l.class_index == r.class_index) && l.score > r.score) ||
                     ((std::fabs(l.score - r.score) <= 1e-6) && l.class_index == r.class_index && l.index < r.index)));
            });
        }
        // in case of "SCORE", pass through, as,
        // it has already gurranteed.
    }

    return selected_boxes;
}

struct SelectedIndex {
    SelectedIndex(int64_t batch_idx, int64_t box_idx, int64_t num_box)
        : flattened_index(batch_idx * num_box + box_idx) {}

    SelectedIndex() = default;

    int64_t flattened_index = 0;
};

struct SelectedOutput {
    SelectedOutput(float class_idx, float score, float x1, float y1, float x2, float y2)
        : class_index{class_idx},
          box_score{score},
          xmin{x1},
          ymin{y1},
          xmax{x2},
          ymax{y2} {}

    SelectedOutput() = default;

    float class_index = 0.0f;
    float box_score = 0.0f;
    float xmin, ymin, xmax, ymax;
};
}  // namespace multiclass_nms_impl

// compute nms over each image and each class
// shared           Y               N
// boxes:       N, M, 4         C, sum(M'), 4
// scores:      N, C, M         C, sum(M')
void multiclass_nms(const float* boxes_data,
                    const Shape& boxes_data_shape,
                    const float* scores_data,
                    const Shape& scores_data_shape,
                    const int64_t* roisnum_data,
                    const Shape& roisnum_data_shape,
                    const op::util::MulticlassNmsBase::Attributes& attrs,
                    float* selected_outputs,
                    const Shape& selected_outputs_shape,
                    int64_t* selected_indices,
                    const Shape& selected_indices_shape,
                    int64_t* valid_outputs) {
    using namespace multiclass_nms_impl;

    const auto shared = scores_data_shape.size() == 3;  // bboxes shared among classes
    const bool has_roinum = roisnum_data;

    std::vector<BoxInfo> filteredBoxes;  // container for the whole batch

    int64_t num_images = 0;
    if (has_roinum) {
        num_images = static_cast<int64_t>(shared ? scores_data_shape[0] : roisnum_data_shape[0]);
    } else {
        OPENVINO_ASSERT(shared, "Expect the input 'scores' is a 3D tensor when there is no 'roisnum' input.");
        num_images = static_cast<int64_t>(scores_data_shape[0]);
    }

    int64_t head = 0;
    for (int64_t i = 0; i < num_images; i++) {
        std::vector<BoxInfo> selected_boxes;
        if (shared) {
            OPENVINO_ASSERT(boxes_data_shape[0] == scores_data_shape[0],
                            "Expect batch size of boxes and scores are the same.");
            OPENVINO_ASSERT(boxes_data_shape[1] == scores_data_shape[2],
                            "Expect box numbers of boxes and scores are the same.");
            const auto num_boxes = boxes_data_shape[1];
            const auto num_classes = scores_data_shape[1];

            const float* boxesPtr = boxes_data + i * num_boxes * 4;
            const float* scoresPtr = scores_data + i * num_classes * num_boxes;

            const Shape boxes_sp = {1, num_boxes, 4};
            const Shape scores_sp = {1, num_classes, num_boxes};

            selected_boxes = multiclass_nms(boxesPtr, boxes_sp, scoresPtr, scores_sp, attrs, i, shared);
        } else {
            if (roisnum_data[i] <= 0) {
                *valid_outputs++ = 0;
                continue;
            }

            OPENVINO_ASSERT(boxes_data_shape[0] == scores_data_shape[0],
                            "Expect class numbers of boxes and scores are the same.");
            OPENVINO_ASSERT(boxes_data_shape[1] == scores_data_shape[1],
                            "Expect box numbers of boxes and scores are the same.");
            const auto num_classes = boxes_data_shape[0];

            const auto boxes = slice_image(boxes_data, boxes_data_shape, head, roisnum_data[i]);
            const auto scores = slice_image(scores_data, scores_data_shape, head, roisnum_data[i]);

            const Shape boxes_sp = {num_classes, static_cast<size_t>(roisnum_data[i]), 4};
            const Shape scores_sp = {num_classes, static_cast<size_t>(roisnum_data[i])};

            selected_boxes = multiclass_nms(boxes.data(), boxes_sp, scores.data(), scores_sp, attrs, i, shared);

            head += roisnum_data[i];
        }

        *valid_outputs++ = static_cast<int64_t>(selected_boxes.size());

        for (auto& v : selected_boxes) {
            filteredBoxes.push_back(v);
        }
    }  // for each batch element

    if (attrs.sort_result_across_batch) { /* sort across batch */
        if (attrs.sort_result_type == op::util::MulticlassNmsBase::SortResultType::SCORE) {
            std::sort(filteredBoxes.begin(), filteredBoxes.end(), [](const BoxInfo& l, const BoxInfo& r) {
                return (l.score > r.score) || (l.score == r.score && l.batch_index < r.batch_index) ||
                       (l.score == r.score && l.batch_index == r.batch_index && l.class_index < r.class_index) ||
                       (l.score == r.score && l.batch_index == r.batch_index && l.class_index == r.class_index &&
                        l.index < r.index);
            });
        } else if (attrs.sort_result_type == op::util::MulticlassNmsBase::SortResultType::CLASSID) {
            std::sort(filteredBoxes.begin(), filteredBoxes.end(), [](const BoxInfo& l, const BoxInfo& r) {
                return (l.class_index < r.class_index) ||
                       (l.class_index == r.class_index && l.batch_index < r.batch_index) ||
                       (l.class_index == r.class_index && l.batch_index == r.batch_index && l.score > r.score) ||
                       (l.class_index == r.class_index && l.batch_index == r.batch_index && l.score == r.score &&
                        l.index < r.index);
            });
        }
    }

    /* output */
    SelectedIndex* selected_indices_ptr = reinterpret_cast<SelectedIndex*>(selected_indices);
    SelectedOutput* selected_scores_ptr = reinterpret_cast<SelectedOutput*>(selected_outputs);

    size_t max_num_of_selected_indices = selected_indices_shape[0];
    size_t output_size = std::min(filteredBoxes.size(), max_num_of_selected_indices);

    size_t idx;
    for (idx = 0; idx < output_size; idx++) {
        const auto& box_info = filteredBoxes[idx];
        SelectedIndex selected_index;
        if (shared) {
            const auto num_boxes = static_cast<int64_t>(boxes_data_shape[1]);
            selected_index = {box_info.batch_index, box_info.index, num_boxes};
        } else {
            const auto num_classes = static_cast<int64_t>(boxes_data_shape[0]);
            int64_t offset = 0;
            for (int64_t i = 0; i < box_info.batch_index; i++) {
                offset += roisnum_data[i];
            }
            selected_index = {(offset + box_info.index), box_info.class_index, num_classes};
        }
        SelectedOutput selected_score{static_cast<float>(box_info.class_index),
                                      box_info.score,
                                      box_info.box.x1,
                                      box_info.box.y1,
                                      box_info.box.x2,
                                      box_info.box.y2};

        selected_indices_ptr[idx] = selected_index;
        selected_scores_ptr[idx] = selected_score;
    }

    SelectedIndex selected_index_filler{0, 0, 0};
    SelectedOutput selected_score_filler{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (; idx < max_num_of_selected_indices; idx++) {
        selected_indices_ptr[idx] = selected_index_filler;
        selected_scores_ptr[idx] = selected_score_filler;
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
