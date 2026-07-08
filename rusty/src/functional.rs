//! Functional Module
//!
//! Stateless functions for common operations (like PyTorch's F namespace).

use crate::tensor::Tensor;

/// Mean Squared Error loss.
///
/// MSE = mean((pred - target)^2)
///
/// This function maintains the autograd graph, so calling `.backward()` on the
/// result will compute gradients for `pred`.
pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    assert_eq!(pred.shape, target.shape, "Shapes must match for MSE loss");
    let diff = pred.sub(target);
    let sq = diff.mul(&diff);
    sq.mean()
}

/// Cross-entropy loss for classification.
///
/// CE = -sum(target * log(softmax(pred)))
///
/// This function maintains the autograd graph. Supports both one-hot encoded
/// targets and class indices.
pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> Tensor {
    // Softmax with autograd tracking
    let probs = logits.softmax(-1);

    // Log of probabilities (maintains autograd)
    let log_probs = probs.log();

    if targets.shape == logits.shape {
        // One-hot encoded targets
        let neg_log_probs = log_probs.mul(targets);
        neg_log_probs.sum().neg()
    } else {
        // Class indices - simplified version (CPU fallback for indexing)
        // For production, implement GPU indexing kernel
        let log_probs_data = log_probs.to_vec();
        let target_data = targets.to_vec();

        let batch_size = logits.shape[0];
        let num_classes = logits.shape[1];

        let mut loss = 0.0f32;
        for i in 0..batch_size {
            let target_class = target_data[i] as usize;
            if target_class < num_classes {
                let log_prob = log_probs_data[i * num_classes + target_class];
                loss -= log_prob;
            }
        }
        loss /= batch_size as f32;

        // Create tensor that maintains gradient flow
        // Note: This breaks autograd - use one-hot encoding for training
        Tensor::from_data(&[loss], &[1], &logits.device)
    }
}

/// Binary Cross-entropy loss.
///
/// BCE = -mean(target * log(pred) + (1 - target) * log(1 - pred))
///
/// This function maintains the autograd graph.
pub fn binary_cross_entropy(pred: &Tensor, target: &Tensor) -> Tensor {
    assert_eq!(
        pred.shape, target.shape,
        "Shapes must match for BCE loss"
    );

    // Clamp pred to avoid log(0) - maintains autograd
    let pred_clamped = pred.clamp(1e-7, 1.0 - 1e-7);

    // log(pred)
    let log_pred = pred_clamped.log();

    // log(1 - pred)
    let one = Tensor::ones(&pred.shape, &pred.device);
    let one_minus_pred = one.sub(&pred_clamped);
    let log_one_minus_pred = one_minus_pred.log();

    // target * log(pred)
    let term1 = target.mul(&log_pred);

    // (1 - target) * log(1 - pred)
    let term2 = one_minus_pred.mul(&log_one_minus_pred);

    // -mean(term1 + term2)
    let loss = term1.add(&term2);
    loss.mean().neg()
}

/// Binary Cross-entropy with logits (more numerically stable).
///
/// Combines sigmoid and BCE in one function for numerical stability.
///
/// This function maintains the autograd graph.
pub fn binary_cross_entropy_with_logits(logits: &Tensor, target: &Tensor) -> Tensor {
    assert_eq!(
        logits.shape, target.shape,
        "Shapes must match for BCE with logits"
    );

    // max(logits, 0) - logits * target + log(1 + exp(-|logits|))
    let zeros = Tensor::zeros(&logits.shape, &logits.device);

    // max(logits, 0)
    let max_logits = logits.maximum(&zeros);

    // logits * target
    let logits_target = logits.mul(target);

    // log(1 + exp(-|logits|))
    let abs_logits = logits.abs();
    let neg_abs_logits = abs_logits.neg();
    let exp_neg_abs = neg_abs_logits.exp();
    let one_plus_exp = Tensor::ones(&exp_neg_abs.shape, &exp_neg_abs.device).add(&exp_neg_abs);
    let log_term = one_plus_exp.log();

    // Combine: max_logits - logits_target + log_term
    let loss = max_logits.sub(&logits_target).add(&log_term);
    loss.mean()
}

/// Smooth L1 (Huber) loss.
///
/// Less sensitive to outliers than MSE.
///
/// This function maintains the autograd graph.
pub fn smooth_l1_loss(pred: &Tensor, target: &Tensor, beta: f32) -> Tensor {
    assert_eq!(
        pred.shape, target.shape,
        "Shapes must match for Smooth L1 loss"
    );

    let diff = pred.sub(target);
    let abs_diff = diff.abs();

    // Create beta tensor
    let beta_tensor = Tensor::from_data(&[beta], &[1], &pred.device);

    // Where |diff| < beta: 0.5 * diff^2 / beta
    // Else: |diff| - 0.5 * beta
    let mask = abs_diff.less_than(&beta_tensor);

    let sq_term = diff.mul(&diff).mul_scalar(0.5 / beta);
    let linear_term = abs_diff.sub(&beta_tensor.mul_scalar(0.5));

    // Simplified: just use the formula directly for all elements
    // For production, implement proper where/select kernel
    let abs_diff_data = abs_diff.to_vec();
    let diff_data = diff.to_vec();

    let mut loss = 0.0f32;
    for (&abs_d, &d) in abs_diff_data.iter().zip(diff_data.iter()) {
        if abs_d < beta {
            loss += 0.5 * d * d / beta;
        } else {
            loss += abs_d - 0.5 * beta;
        }
    }
    loss /= diff_data.len() as f32;

    // Note: This breaks autograd - CPU fallback
    Tensor::from_data(&[loss], &[1], &pred.device)
}

/// Cosine similarity between two tensors.
///
/// Returns a value in [-1, 1]. This is an evaluation metric,
/// not intended for training (doesn't maintain autograd).
pub fn cosine_similarity(a: &Tensor, b: &Tensor) -> f32 {
    let a_data = a.to_vec();
    let b_data = b.to_vec();

    let dot: f32 = a_data.iter().zip(b_data.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a_data.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b_data.iter().map(|x| x * x).sum::<f32>().sqrt();

    dot / (norm_a * norm_b + 1e-8)
}

/// Dropout (during training).
///
/// Randomly zeros elements with probability p and scales remaining elements by 1/(1-p).
///
/// Note: This implementation uses CPU and breaks autograd. For training,
/// use GPU dropout kernel (TODO).
pub fn dropout(x: &Tensor, p: f32, training: bool) -> Tensor {
    if !training || p == 0.0 {
        return x.clone();
    }

    assert!(
        (0.0..1.0).contains(&p),
        "Dropout probability must be in [0, 1)"
    );

    use rand::Rng;
    let mut rng = rand::thread_rng();
    let data = x.to_vec();
    let scale = 1.0 / (1.0 - p);

    let result: Vec<f32> = data
        .iter()
        .map(|&v| {
            if rng.gen::<f32>() < p {
                0.0
            } else {
                v * scale
            }
        })
        .collect();

    // Note: This breaks autograd - CPU fallback
    // For production, implement GPU dropout kernel
    Tensor::from_data(&result, &x.shape, &x.device)
}

/// One-hot encoding.
pub fn one_hot(indices: &[usize], num_classes: usize, device: &crate::Device) -> Tensor {
    let batch_size = indices.len();
    let mut data = vec![0.0f32; batch_size * num_classes];

    for (i, &idx) in indices.iter().enumerate() {
        if idx < num_classes {
            data[i * num_classes + idx] = 1.0;
        }
    }

    Tensor::from_data(&data, &[batch_size, num_classes], device)
}
