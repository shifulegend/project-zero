#include "core/platform.h"
#include "memory/aligned_alloc.h"
#include "test_harness.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * Audit Test: Mem-01 (Zero Size Allocation)
 * Ensures aligned_alloc safely returns NULL or a safe 0-byte pointer without
 * invoking UB in posix_memalign or crashing the custom wrappers.
 */
static void aud_mem_zero_size(void) {
  void *ptr = tn_aligned_alloc(0, TN_SIMD_ALIGN);
  void *ptr2 = tn_aligned_calloc(0, sizeof(float), TN_SIMD_ALIGN);
  void *ptr3 = tn_aligned_calloc(10, 0, TN_SIMD_ALIGN);

  TEST_ASSERT(ptr == NULL, "tn_aligned_alloc(0) returns NULL");
  TEST_ASSERT(ptr2 == NULL, "tn_aligned_calloc(0, size) returns NULL");
  TEST_ASSERT(ptr3 == NULL, "tn_aligned_calloc(count, 0) returns NULL");
}

/**
 * Audit Test: Mem-02 (Invalid Alignment)
 * posix_memalign requires alignment to be a power of two multiple of void*.
 * Passing an invalid alignment like 3 or 10 should gracefully fail.
 */
static void aud_mem_invalid_align(void) {
  void *ptr1 = tn_aligned_alloc(1024, 3);
  void *ptr2 = tn_aligned_alloc(1024, 10);
  void *ptr3 =
      tn_aligned_alloc(1024, sizeof(void *) / 2); // Less than pointer size

  TEST_ASSERT(ptr1 == NULL, "Invalid alignment (3) returns NULL");
  TEST_ASSERT(ptr2 == NULL, "Invalid alignment (10) returns NULL");
  TEST_ASSERT(ptr3 == NULL, "Invalid alignment (< ptr_size) returns NULL");
}

/**
 * Audit Test: Mem-03 (Calloc Integer Overflow)
 * Passing gigantic numbers to calloc should trigger the internal overflow check
 * rather than attempting to allocate a tiny wrapped-around amount.
 */
static void aud_mem_calloc_overflow(void) {
  size_t gigantic_count = SIZE_MAX / 2 + 100;
  size_t size = 4;

  // gigantic_count * 4 > SIZE_MAX, should wrap around to a small number
  void *ptr = tn_aligned_calloc(gigantic_count, size, TN_SIMD_ALIGN);
  TEST_ASSERT(ptr == NULL, "tn_aligned_calloc traps integer overflow");
}

int main(void) {
  RUN_TEST(aud_mem_zero_size);
  RUN_TEST(aud_mem_invalid_align);
  RUN_TEST(aud_mem_calloc_overflow);
  TEST_SUMMARY();
}
