#pragma once

#include "model.h"

struct Sampler {
  int vocab_size;

  Sampler(const Config& config);

  // Return the probability score corresponding to `logits[index]`.
  // This is equivalent to taking the softmax of the logits and returning
  // the value at index `index`.
  float sample_prob(int index, const InferenceState& s);
  // Return the index of the maximum value in `logits`.
  int sample_argmax(const InferenceState& s);
};