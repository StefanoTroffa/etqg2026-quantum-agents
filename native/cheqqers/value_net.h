// A small value network (multi-layer perceptron, ReLU) as an optional leaf evaluation of the search.
//
// Trained by src/training/train_cheqqers_value_mlp.py; its JSON is read in Python (native_cheqqers.ValueNet) and the
// layers are passed here as flat arrays (cq_net_new). Input = encode_position(record) flattened
// (src/models/cheqqers_policy_value.py): per square [one-hot none / own man / own queen / enemy man / enemy queen
// relative to the side to move, occupied probability, superposition flag], then eval_features for the side to move,
// then the multi-jump flag. Output: one logit, "the side to move wins".
//
// Without a network (SearchOpts::net == nullptr) the search is unchanged: the hook is a single pointer test in leaf().
#pragma once

#include <vector>

#include "engine.h"

namespace cq {

class ValueNet {
   public:
    // dims[0] = inputs, dims[k] = outputs of layer k (the last must be 1). params: per layer the weights row-major
    // [out][in] (PyTorch's nn.Linear layout), then the biases [out].
    ValueNet(const std::vector<int>& dims, const double* params, long long nparams);

    int inputs() const { return dims_[0]; }
    // The network's input for this position (side to move's view); x must hold inputs() doubles.
    void encode(Env& env, double* x) const;
    // Forward pass on a given input; returns the logit.
    double forward(const double* x) const;
    // encode + forward
    double logit(Env& env) const;
    // The leaf value in the linear evaluation's squash scale, from `persp`'s point of view: 900 tanh(logit / 2) for
    // the side to move, negated for the other side. Terminal positions are not handled here (the caller keeps
    // Env::evaluate's +-1000 / 0).
    // Residual mode (set_residual): logit = w.logit_per_man * (w . features) + lambda * net, i.e. the fitted linear
    // evaluation's logit plus a learned correction (the net was trained with the linear logit as a fixed offset).
    // lambda = 0 gives the linear evaluation's value (the features are antisymmetric in the point of view).
    double value(Env& env, int persp, const Weights& w) const;
    void set_residual(double lambda) {
        residual_ = true;
        lambda_ = lambda;
    }
    bool residual() const { return residual_; }

   private:
    bool residual_ = false;
    double lambda_ = 1.0;
    std::vector<int> dims_;
    std::vector<std::vector<double>> wt_;  // per layer, transposed: [in][out]
    std::vector<std::vector<double>> b_;
};

// The network input (7 per square + NFEAT features + multi-jump flag) of any position, side to move's view; x must
// hold env.N * 7 + NFEAT + 1 doubles. Used by ValueNet::encode and by the search's leaf sampler (search.h LeafSink).
void encode_position(Env& env, double* x);

}  // namespace cq
