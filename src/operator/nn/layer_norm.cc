/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file layer_norm.cc
 * \brief Implements Ba et. al, Layer Normalization (https://arxiv.org/abs/1607.06450).
*/

#include "layer_norm-inl.h"
#include <nnvm/op_attr_types.h>
#include "../elemwise_op_common.h"
#include "layer_norm_cpu.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(LayerNormParam);

static bool LayerNormShape(const nnvm::NodeAttrs& attrs,
                           mxnet::ShapeVector *in_shape,
                           mxnet::ShapeVector *out_shape) {
  const LayerNormParam& param = nnvm::get<LayerNormParam>(attrs.parsed);
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), 3U) << "Input:[data, gamma, beta]";
  const mxnet::TShape &dshape = in_shape->at(layernorm::kData);
  if (!mxnet::ndim_is_known(dshape)) {
    return false;
  }

  int axis = GetRealAxis(param.axis, dshape.ndim());
  CHECK(axis >= 0 && axis < dshape.ndim())
    << "Channel axis out of range: axis=" << param.axis;

  const index_t channelCount = dshape[axis];

  SHAPE_ASSIGN_CHECK(*in_shape,
                     layernorm::kGamma,
                     mxnet::TShape(Shape1(channelCount)));
  SHAPE_ASSIGN_CHECK(*in_shape,
                     layernorm::kBeta,
                     mxnet::TShape(Shape1(channelCount)));
  out_shape->clear();
  out_shape->push_back(dshape);                // kOut
  mxnet::TShape moments_shape(dshape.begin(), dshape.end());
  moments_shape[axis] = 1;
  out_shape->push_back(moments_shape);  // kMean
  out_shape->push_back(moments_shape);  // kInvstd
  return true;
}

/* Wrap the above LayerNormCPUKernel in MXNet's API.  Returns true if it
 * is able to run.
 */
bool LayerNormCPU(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx, const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  const LayerNormParam& param = nnvm::get<LayerNormParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 3U);

  switch (req[layernorm::kOut]) {
    case kNullOp:
      return true;
    case kWriteTo:
      break;
    case kWriteInplace:
      break;
    default:
      // Should only be kAddTo, which isn't supported by the others implementation either.
      return false;
  }
  // Axis must be the last one.
  int axis = GetRealAxis(param.axis, inputs[layernorm::kData].ndim());
  if (axis != inputs[layernorm::kData].ndim() - 1) {
    return false;
  }
  MSHADOW_REAL_TYPE_SWITCH(inputs[layernorm::kData].type_flag_, DType, {
    LayerNormCPUKernel<DType>(
        inputs[layernorm::kData].shape_[axis],
        outputs[layernorm::kMean].Size(),
        param.eps,
        inputs[layernorm::kData].dptr<DType>(),
        inputs[layernorm::kGamma].dptr<DType>(),
        inputs[layernorm::kBeta].dptr<DType>(),
        outputs[layernorm::kOut].dptr<DType>(),
        outputs[layernorm::kMean].dptr<DType>(),
        outputs[layernorm::kStd].dptr<DType>());
  });
  return true;
}

#if MSHADOW_USE_MKL == 1 && MXNET_USE_MKL_LAYERNORM == 1
bool LayerNormComputeMKL(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  const LayerNormParam& param = nnvm::get<LayerNormParam>(attrs.parsed);
  if (req[0] == kNullOp) return true;
  CHECK_NE(req[0], kAddTo);
  CHECK_EQ(inputs.size(), 3U);
  int axis = GetRealAxis(param.axis, inputs[0].ndim());

  // This optimization only applys for LayerNorm on the last dimension with dtype FP32 or FP64.
  if (axis == (inputs[layernorm::kData].ndim() - 1) &&
      (inputs[0].type_flag_ == kFloat32 || inputs[0].type_flag_ == kFloat64)) {
    // Compute necessary data for the reduce operation.
    mxnet::TShape red_src_shape, red_dst_shape;
    BroadcastReduceShapeCompact(inputs[layernorm::kData].shape_, outputs[layernorm::kMean].shape_,
                                &red_src_shape, &red_dst_shape);
    const TBlob in_data = inputs[layernorm::kData].reshape(red_src_shape);
    const TBlob mean_data = outputs[layernorm::kMean].reshape(red_dst_shape);
    const TBlob std_data = outputs[layernorm::kStd].reshape(red_dst_shape);
    const int outter_size = red_dst_shape.Size();
    const int channel_size = red_src_shape.Size() / red_dst_shape.Size();

    // call
    MSHADOW_SGL_DBL_TYPE_SWITCH(in_data.type_flag_, DType, {
      mkl_func::LayerNormLastDim(outter_size, channel_size,
                                 in_data.dptr<DType>(),
                                 outputs[layernorm::kOut].dptr<DType>(),
                                 inputs[layernorm::kGamma].dptr<DType>(),
                                 inputs[layernorm::kBeta].dptr<DType>(),
                                 outputs[layernorm::kMean].dptr<DType>(),
                                 outputs[layernorm::kStd].dptr<DType>(),
                                 static_cast<DType>(param.eps));
    });
    return true;
  } else {
    // fallback
    return false;
  }
}
#endif

template<>
void LayerNormCompute<cpu>(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx, const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
#if MSHADOW_USE_MKL == 1 && MXNET_USE_MKL_LAYERNORM == 1
  if (LayerNormComputeMKL(attrs, ctx, inputs, req, outputs)) return;
#endif
  if (LayerNormCPU(attrs, ctx, inputs, req, outputs)) return;
  LayerNormComputeGeneral<cpu>(attrs, ctx, inputs, req, outputs);
}

template<>
void LayerNormGradCompute<cpu>(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx, const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  return LayerNormGradComputeGeneral<cpu>(attrs, ctx, inputs, req, outputs);
}

NNVM_REGISTER_OP(LayerNorm)
.add_alias("_npx_layer_norm")
.describe(R"code(Layer normalization.

Normalizes the channels of the input tensor by mean and variance, and applies a scale ``gamma`` as
well as offset ``beta``.

Assume the input has more than one dimension and we normalize along axis 1.
We first compute the mean and variance along this axis and then 
compute the normalized output, which has the same shape as input, as following:

.. math::

  out = \frac{data - mean(data, axis)}{\sqrt{var(data, axis) + \epsilon}} * gamma + beta

Both ``gamma`` and ``beta`` are learnable parameters.

Unlike BatchNorm and InstanceNorm,  the *mean* and *var* are computed along the channel dimension.

Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and
``data_std``. Note that no gradient will be passed through these two outputs.

The parameter ``axis`` specifies which axis of the input shape denotes
the 'channel' (separately normalized groups).  The default is -1, which sets the channel
axis to be the last item in the input shape.

)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(3)
.set_attr_parser(ParamParser<LayerNormParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data", "gamma", "beta"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output", "mean", "std"};
})
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
    [](const NodeAttrs& attrs) {
  const LayerNormParam& param = nnvm::get<LayerNormParam>(attrs.parsed);
  return param.output_mean_var ? 3 : 1;
})
.set_attr<mxnet::FInferShape>("FInferShape", LayerNormShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<3, 3>)
.set_attr<FCompute>("FCompute<cpu>", LayerNormCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", [](const nnvm::ObjectPtr& n,
                                           const std::vector<nnvm::NodeEntry>& ograds) {
  std::vector<nnvm::NodeEntry> heads;
  heads.push_back(ograds[0]);  // ograd
  heads.push_back(n->inputs[0]);  // data
  heads.push_back(n->inputs[1]);  // gamma
  heads.emplace_back(n, 1, 0);  // mean
  heads.emplace_back(n, 2, 0);  // std
  return MakeGradNode("_backward_LayerNorm", n, heads, n->attrs.dict);
})
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
  return std::vector<std::pair<int, int> >{{0, 0}};
})
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.add_argument("data", "NDArray-or-Symbol", "Input data to layer normalization")
.add_argument("gamma", "NDArray-or-Symbol", "gamma array")
.add_argument("beta", "NDArray-or-Symbol", "beta array")
.add_arguments(LayerNormParam::__FIELDS__());


NNVM_REGISTER_OP(_backward_LayerNorm)
.set_num_inputs(5)
.set_num_outputs(3)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(ParamParser<LayerNormParam>)
.set_attr<FCompute>("FCompute<cpu>", LayerNormGradCompute<cpu>)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
});

}  // namespace op
}  // namespace mxnet
