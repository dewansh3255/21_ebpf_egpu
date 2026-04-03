// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
#define _GNU_SOURCE
#include <dlfcn.h>
#include <signal.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <sys/resource.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include "./.output/kernelretsnoop.skel.h"
#include <inttypes.h>

#define warn(...) fprintf(stderr, __VA_ARGS__)
#define MAX_THREADS 1024  /* track entry timestamps for up to this many threads */

static int libbpf_print_fn(enum libbpf_print_level level,
			   const char *format, va_list args)
{
	return vfprintf(stderr, format, args);
}

static volatile bool exiting = false;
static void sig_handler(int sig) { exiting = true; }

struct data {
	uint64_t x, y, z;
	uint64_t timestamp;
	uint64_t type;   /* 0=entry, 1=exit */
};

/*
 * Simple flat array to store the last entry timestamp per thread.
 * Keyed by flat thread id = x (for vec_add which only uses x dimension).
 * For a real workload you would use a hash map keyed by (x,y,z,blockx,blocky).
 */
static uint64_t entry_ts[MAX_THREADS];

struct state {
	uint64_t count;
};

static void poll_callback(const void *data, uint64_t size, void *ctx)
{
	struct state *state = (struct state *)ctx;
	const struct data *ev = data;

	if (ev->type == 0) {
		/* ENTRY: store timestamp, print start line */
		uint64_t tid = ev->x;
		if (tid < MAX_THREADS)
			entry_ts[tid] = ev->timestamp;

		printf("[ENTRY] Thread (%lu, %lu, %lu)  ts=%lu\n",
		       ev->x, ev->y, ev->z, ev->timestamp);
	} else {
		/* EXIT: print exit line + duration if we have the entry */
		uint64_t tid = ev->x;
		uint64_t entry = (tid < MAX_THREADS) ? entry_ts[tid] : 0;
		uint64_t duration = (entry > 0 && ev->timestamp >= entry)
				? (ev->timestamp - entry) : 0;

		if (duration > 0)
			printf("[EXIT ] Thread (%lu, %lu, %lu)  ts=%lu  duration=%lu ns\n",
			       ev->x, ev->y, ev->z, ev->timestamp, duration);
		else
			printf("[EXIT ] Thread (%lu, %lu, %lu)  ts=%lu\n",
			       ev->x, ev->y, ev->z, ev->timestamp);
	}

	state->count += 1;
}

int main(int argc, char **argv)
{
	struct kernelretsnoop_bpf *skel;
	int err;

	libbpf_set_print(libbpf_print_fn);
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	memset(entry_ts, 0, sizeof(entry_ts));

	skel = kernelretsnoop_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open BPF skeleton\n");
		return 1;
	}

	err = kernelretsnoop_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load BPF skeleton\n");
		goto cleanup;
	}

	err = kernelretsnoop_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "Failed to attach BPF skeleton\n");
		goto cleanup;
	}

	int (*poll_fn)(int, void *, void (*)(const void *, uint64_t, void *)) =
		dlsym(RTLD_DEFAULT,
		      "bpftime_syscall_server__poll_gpu_ringbuf_map");
	if (!poll_fn) {
		puts("This example can only be used under bpftime!");
		goto cleanup;
	}

	int mapfd = bpf_map__fd(skel->maps.rb);
	struct state state = {0};

	printf("%-8s  %-30s  %-20s  %s\n",
	       "EVENT", "THREAD (x,y,z)", "TIMESTAMP (ns)", "DURATION (ns)");
	printf("----------------------------------------------------------------------\n");

	while (!exiting) {
		sleep(1);
		err = poll_fn(mapfd, &state, poll_callback);
		if (err < 0) {
			printf("Poll error: %d\n", err);
			goto cleanup;
		}
		if (state.count > 0)
			printf("--- total events so far: %lu ---\n", state.count);
	}

cleanup:
	kernelretsnoop_bpf__destroy(skel);
	return err < 0 ? -err : 0;
}
