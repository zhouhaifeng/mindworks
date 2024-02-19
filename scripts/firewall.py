from bcc import BPF
from socket import inet_aton

# BPF program in C
program = """
#include <uapi/linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>

int block_tcp_port(struct __sk_buff *skb) {
    struct ethhdr *eth = bpf_hdr_pointer(skb);
    struct iphdr *ip = (struct iphdr *)(eth + 1);
    struct tcphdr *tcp = (struct tcphdr *)(ip + 1);

    // Define the TCP port you want to block (e.g., port 80)
    uint16_t blocked_port = 80;

    if (ip->protocol == IPPROTO_TCP && tcp->dest == htons(blocked_port)) {
        // Drop the packet
        return XDP_DROP;
    }

    return XDP_PASS;
}
"""

firewall_program = """
#include <uapi/linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>

int firewall(struct __sk_buff *skb) {
    struct ethhdr *eth = 0;
    struct iphdr *ip = 0;
    struct tcphdr *tcp = 0;
    struct udphdr *udp = 0;

    if (skb == 0)
    {
        return XDP_DROP;
    }

    eth = bpf_hdr_pointer(skb);
    if (eth == 0)
    {
        return XDP_DROP;
    }

    ip = (struct iphdr *)(eth + 1);
    if (ip == 0)
    {
        return XDP_DROP;
    }

    if (ip->protocol == IPPROTO_TCP) 
    {
        tcp = (struct tcphdr *)(ip + 1);
        if (tcp == 0)
        {
            return XDP_DROP;
        }
    }

    if (ip->protocol == IPPROTO_UDP) 
    {
        udp = (struct udphdr *)(ip + 1);
        if (udp == 0)
        {
            return XDP_DROP;
        }
    }

    if ((ip->protocol == IPPROTO_TCP) && (tcp->dest == htons(80))
        || ((ip->protocol == IPPROTO_UDP) && (udp->dest == htons(25))) {
        // Drop the packet
        return XDP_PASS;
    }

    return XDP_PASS;
}
"""


# Load BPF program
bpf = BPF(text=firewall_program)

# Attach the program to a network interface (e.g., eth0)
bpf.attach_xdp(device="wlp58s0", fn_name="firewall")

try:
    print("pass tcp 80 and udp 25. Press Ctrl+C to stop.")
    bpf.trace_print()
except KeyboardInterrupt:
    pass

# Detach the program when done
#bpf.remove_xdp(device="wlp58s0")