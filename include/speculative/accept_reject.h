#ifndef TN_SPECULATIVE_ACCEPT_REJECT_H
#define TN_SPECULATIVE_ACCEPT_REJECT_H

/**
 * Phase 18 (speculative decoding), Stage 5: rejection-sampling math shared
 * by speculative_generate()/speculative_generate_with_callback().
 *
 * Greedy (temperature <= 0.0f || temperature < 1e-6f, matching
 * generate_with_callback()'s existing threshold exactly): accept
 * draft_tokens[i] iff it equals argmax(verify_check_logits[i]); on the
 * first mismatch, emit the verifier's own argmax at that position instead.
 * When every drafted token is accepted, emit argmax(bonus_logits) as a
 * bonus token (the verifier already computed it for free as part of the
 * same batched pass). Because transformer_forward_batch() is
 * teacher-forcing-equivalent to sequential transformer_forward() calls,
 * this construction makes greedy speculative output IDENTICAL to plain
 * greedy generation, token for token -- by construction, not by separate
 * proof.
 *
 * Stochastic (temperature > 0): accept draft_tokens[i] with probability
 * min(1, p_verifier(x)/p_draft(x)) where p_* are softmax(logits/temperature);
 * on reject, resample from the renormalized residual
 * max(0, p_verifier - p_draft). When every drafted token is accepted, the
 * bonus token is sampled from softmax(bonus_logits/temperature).
 */

/**
 * @param draft_tokens        spec_length token IDs the draft model produced.
 * @param draft_logits        [spec_length][vocab_size] row-major, row i is
 *                            the draft model's own distribution
 *                            draft_tokens[i] was sampled from. Only read in
 *                            stochastic mode (temperature > 0) -- may be
 *                            NULL in greedy mode.
 * @param verify_check_logits [spec_length][vocab_size] row-major, row i is
 *                            the verifier's prediction made BEFORE
 *                            draft_tokens[i] was fed -- i.e. row i-1 of
 *                            transformer_forward_batch()'s output (or the
 *                            pre-round carried-over prediction for i=0),
 *                            NOT row i (see transformer/forward.h's row
 *                            convention: row i predicts the token AFTER
 *                            tokens[0..i], not before tokens[i]). Building
 *                            this shifted array is the caller's job
 *                            (speculative_generate_with_callback()) so the
 *                            off-by-one is never implicit here.
 * @param bonus_logits         Single row: the verifier's prediction for the
 *                            position one past the last drafted token
 *                            (transformer_forward_batch()'s row
 *                            spec_length-1) -- used only when every drafted
 *                            token is accepted.
 * @param rng_state           Only consulted in stochastic mode.
 * @param n_accept            Out: number of accepted draft tokens, 0..spec_length.
 * @param emit_token          Out: exactly one additional token to emit after
 *                            the *n_accept accepted draft tokens -- either a
 *                            resampled token (on reject) or the bonus token
 *                            (after full acceptance).
 */
void accept_reject_round(const int *draft_tokens, const float *draft_logits,
                          const float *verify_check_logits, const float *bonus_logits,
                          int spec_length, int vocab_size, float temperature,
                          unsigned long long *rng_state,
                          int *n_accept, int *emit_token);

#endif /* TN_SPECULATIVE_ACCEPT_REJECT_H */
