
util.global private @token_embd_weight = #stream.parameter.named<"weights"::"token_embd.weight"> : tensor<32x64xf32>
util.global private @blk0_attn_norm_weight = #stream.parameter.named<"weights"::"blk.0.attn_norm.weight"> : tensor<64xf32>
util.global private @blk0_attn_q_weight = #stream.parameter.named<"weights"::"blk.0.attn_q.weight"> : tensor<64x64xf32>
util.global private @blk0_attn_k_weight = #stream.parameter.named<"weights"::"blk.0.attn_k.weight"> : tensor<64x64xf32>
util.global private @blk0_attn_v_weight = #stream.parameter.named<"weights"::"blk.0.attn_v.weight"> : tensor<64x64xf32>
util.global private @blk0_attn_output_weight = #stream.parameter.named<"weights"::"blk.0.attn_output.weight"> : tensor<64x64xf32>
util.global private @blk0_ffn_norm_weight = #stream.parameter.named<"weights"::"blk.0.ffn_norm.weight"> : tensor<64xf32>
util.global private @blk0_ffn_gate_weight = #stream.parameter.named<"weights"::"blk.0.ffn_gate.weight"> : tensor<128x64xf32>
util.global private @blk0_ffn_up_weight = #stream.parameter.named<"weights"::"blk.0.ffn_up.weight"> : tensor<128x64xf32>
util.global private @blk0_ffn_down_weight = #stream.parameter.named<"weights"::"blk.0.ffn_down.weight"> : tensor<64x128xf32>
util.global private @output_norm_weight = #stream.parameter.named<"weights"::"output_norm.weight"> : tensor<64xf32>
util.global private @output_weight = #stream.parameter.named<"weights"::"output.weight"> : tensor<32x64xf32>

func.func private @rms_norm(%x: tensor<4x64xf32>, %w: tensor<64xf32>) -> tensor<4x64xf32> {
  %c0 = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %eps = arith.constant 1.000000e-05 : f32
  %c64 = arith.constant 6.400000e+01 : f32
  %sq = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
    ins(%x : tensor<4x64xf32>) outs(%x : tensor<4x64xf32>) {
    ^bb0(%a: f32, %b: f32):
      %p = arith.mulf %a, %a : f32
      linalg.yield %p : f32
  } -> tensor<4x64xf32>
  %init = tensor.empty() : tensor<4xf32>
  %z = linalg.fill ins(%c0 : f32) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %ms = linalg.reduce ins(%sq : tensor<4x64xf32>) outs(%z : tensor<4xf32>) dimensions = [1]
    (%in: f32, %acc: f32) {
      %s = arith.addf %in, %acc : f32
      linalg.yield %s : f32
    }
  %inv = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
    ins(%ms : tensor<4xf32>) outs(%ms : tensor<4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %m = arith.divf %a, %c64 : f32
      %meps = arith.addf %m, %eps : f32
      %root = math.sqrt %meps : f32
      %i = arith.divf %one, %root : f32
      linalg.yield %i : f32
  } -> tensor<4xf32>
  %y = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
    ins(%x, %inv, %w : tensor<4x64xf32>, tensor<4xf32>, tensor<64xf32>) outs(%x : tensor<4x64xf32>) {
    ^bb0(%a: f32, %i: f32, %ww: f32, %o: f32):
      %t = arith.mulf %a, %i : f32
      %r = arith.mulf %t, %ww : f32
      linalg.yield %r : f32
  } -> tensor<4x64xf32>
  return %y : tensor<4x64xf32>
}

func.func private @linear64(%x: tensor<4x64xf32>, %w: tensor<64x64xf32>) -> tensor<4x64xf32> {
  %c0 = arith.constant 0.0 : f32
  %ti = tensor.empty() : tensor<64x64xf32>
  %wt = linalg.transpose ins(%w : tensor<64x64xf32>) outs(%ti : tensor<64x64xf32>) permutation = [1, 0]
  %init = tensor.empty() : tensor<4x64xf32>
  %z = linalg.fill ins(%c0 : f32) outs(%init : tensor<4x64xf32>) -> tensor<4x64xf32>
  %y = linalg.matmul ins(%x, %wt : tensor<4x64xf32>, tensor<64x64xf32>) outs(%z : tensor<4x64xf32>) -> tensor<4x64xf32>
  return %y : tensor<4x64xf32>
}

func.func private @linear_ffn_up(%x: tensor<4x64xf32>, %w: tensor<128x64xf32>) -> tensor<4x128xf32> {
  %c0 = arith.constant 0.0 : f32
  %ti = tensor.empty() : tensor<64x128xf32>
  %wt = linalg.transpose ins(%w : tensor<128x64xf32>) outs(%ti : tensor<64x128xf32>) permutation = [1, 0]
  %init = tensor.empty() : tensor<4x128xf32>
  %z = linalg.fill ins(%c0 : f32) outs(%init : tensor<4x128xf32>) -> tensor<4x128xf32>
  %y = linalg.matmul ins(%x, %wt : tensor<4x64xf32>, tensor<64x128xf32>) outs(%z : tensor<4x128xf32>) -> tensor<4x128xf32>
  return %y : tensor<4x128xf32>
}

func.func private @linear_ffn_down(%x: tensor<4x128xf32>, %w: tensor<64x128xf32>) -> tensor<4x64xf32> {
  %c0 = arith.constant 0.0 : f32
  %ti = tensor.empty() : tensor<128x64xf32>
  %wt = linalg.transpose ins(%w : tensor<64x128xf32>) outs(%ti : tensor<128x64xf32>) permutation = [1, 0]
  %init = tensor.empty() : tensor<4x64xf32>
  %z = linalg.fill ins(%c0 : f32) outs(%init : tensor<4x64xf32>) -> tensor<4x64xf32>
  %y = linalg.matmul ins(%x, %wt : tensor<4x128xf32>, tensor<128x64xf32>) outs(%z : tensor<4x64xf32>) -> tensor<4x64xf32>
  return %y : tensor<4x64xf32>
}

func.func @prefill(%tokens: tensor<4xi64>) -> tensor<32xf32> {
  %emb_t = util.global.load @token_embd_weight : tensor<32x64xf32>
  %attn_nw = util.global.load @blk0_attn_norm_weight : tensor<64xf32>
  %wq = util.global.load @blk0_attn_q_weight : tensor<64x64xf32>
  %wk = util.global.load @blk0_attn_k_weight : tensor<64x64xf32>
  %wv = util.global.load @blk0_attn_v_weight : tensor<64x64xf32>
  %wo = util.global.load @blk0_attn_output_weight : tensor<64x64xf32>
  %ffn_nw = util.global.load @blk0_ffn_norm_weight : tensor<64xf32>
  %wgate = util.global.load @blk0_ffn_gate_weight : tensor<128x64xf32>
  %wup = util.global.load @blk0_ffn_up_weight : tensor<128x64xf32>
  %wdown = util.global.load @blk0_ffn_down_weight : tensor<64x128xf32>
  %out_nw = util.global.load @output_norm_weight : tensor<64xf32>
  %wout = util.global.load @output_weight : tensor<32x64xf32>

  %c0i = arith.constant 0 : index
  %c1i = arith.constant 1 : index
  %c2i = arith.constant 2 : index
  %c3i = arith.constant 3 : index
  %t0 = tensor.extract %tokens[%c0i] : tensor<4xi64>
  %t1 = tensor.extract %tokens[%c1i] : tensor<4xi64>
  %t2 = tensor.extract %tokens[%c2i] : tensor<4xi64>
  %t3 = tensor.extract %tokens[%c3i] : tensor<4xi64>
  %i0 = arith.index_cast %t0 : i64 to index
  %i1 = arith.index_cast %t1 : i64 to index
  %i2 = arith.index_cast %t2 : i64 to index
  %i3 = arith.index_cast %t3 : i64 to index
  %r0 = tensor.extract_slice %emb_t[%i0, 0] [1, 64] [1, 1] : tensor<32x64xf32> to tensor<1x64xf32>
  %r1 = tensor.extract_slice %emb_t[%i1, 0] [1, 64] [1, 1] : tensor<32x64xf32> to tensor<1x64xf32>
  %r2 = tensor.extract_slice %emb_t[%i2, 0] [1, 64] [1, 1] : tensor<32x64xf32> to tensor<1x64xf32>
  %r3 = tensor.extract_slice %emb_t[%i3, 0] [1, 64] [1, 1] : tensor<32x64xf32> to tensor<1x64xf32>
  %e0 = tensor.empty() : tensor<4x64xf32>
  %e1 = tensor.insert_slice %r0 into %e0[0, 0] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<4x64xf32>
  %e2 = tensor.insert_slice %r1 into %e1[1, 0] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<4x64xf32>
  %e3 = tensor.insert_slice %r2 into %e2[2, 0] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<4x64xf32>
  %h_in = tensor.insert_slice %r3 into %e3[3, 0] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<4x64xf32>

  %xn = func.call @rms_norm(%h_in, %attn_nw) : (tensor<4x64xf32>, tensor<64xf32>) -> tensor<4x64xf32>
  %q = func.call @linear64(%xn, %wq) : (tensor<4x64xf32>, tensor<64x64xf32>) -> tensor<4x64xf32>
  %k = func.call @linear64(%xn, %wk) : (tensor<4x64xf32>, tensor<64x64xf32>) -> tensor<4x64xf32>
  %v = func.call @linear64(%xn, %wv) : (tensor<4x64xf32>, tensor<64x64xf32>) -> tensor<4x64xf32>

  // [S,H] -> [S,NH,D] -> [NH,S,D]
  %q3 = tensor.expand_shape %q [[0], [1, 2]] output_shape [4, 4, 16] : tensor<4x64xf32> into tensor<4x4x16xf32>
  %k3 = tensor.expand_shape %k [[0], [1, 2]] output_shape [4, 4, 16] : tensor<4x64xf32> into tensor<4x4x16xf32>
  %v3 = tensor.expand_shape %v [[0], [1, 2]] output_shape [4, 4, 16] : tensor<4x64xf32> into tensor<4x4x16xf32>
  %q_ti = tensor.empty() : tensor<4x4x16xf32>
  %k_ti = tensor.empty() : tensor<4x4x16xf32>
  %v_ti = tensor.empty() : tensor<4x4x16xf32>
  %qb = linalg.transpose ins(%q3 : tensor<4x4x16xf32>) outs(%q_ti : tensor<4x4x16xf32>) permutation = [1, 0, 2]
  %kb = linalg.transpose ins(%k3 : tensor<4x4x16xf32>) outs(%k_ti : tensor<4x4x16xf32>) permutation = [1, 0, 2]
  %vb = linalg.transpose ins(%v3 : tensor<4x4x16xf32>) outs(%v_ti : tensor<4x4x16xf32>) permutation = [1, 0, 2]

  // K^T : [NH,D,S]
  %kt_i = tensor.empty() : tensor<4x16x4xf32>
  %kt = linalg.transpose ins(%kb : tensor<4x4x16xf32>) outs(%kt_i : tensor<4x16x4xf32>) permutation = [0, 2, 1]

  %c0 = arith.constant 0.0 : f32
  %neg = arith.constant -1.0e+30 : f32
  %scale = arith.constant 2.5e-01 : f32
  %sc_i = tensor.empty() : tensor<4x4x4xf32>
  %sc_z = linalg.fill ins(%c0 : f32) outs(%sc_i : tensor<4x4x4xf32>) -> tensor<4x4x4xf32>
  %scores = linalg.batch_matmul ins(%qb, %kt : tensor<4x4x16xf32>, tensor<4x16x4xf32>) outs(%sc_z : tensor<4x4x4xf32>) -> tensor<4x4x4xf32>
  %scores_s = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%scores : tensor<4x4x4xf32>) outs(%scores : tensor<4x4x4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %m = arith.mulf %a, %scale : f32
      linalg.yield %m : f32
  } -> tensor<4x4x4xf32>
  // causal mask
  %masked = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%scores_s : tensor<4x4x4xf32>) outs(%scores_s : tensor<4x4x4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %i = linalg.index 1 : index
      %j = linalg.index 2 : index
      %cmp = arith.cmpi sgt, %j, %i : index
      %sel = arith.select %cmp, %neg, %a : f32
      linalg.yield %sel : f32
  } -> tensor<4x4x4xf32>
  %sm_i = tensor.empty() : tensor<4x4x4xf32>
  %attn = linalg.softmax dimension(2) ins(%masked : tensor<4x4x4xf32>) outs(%sm_i : tensor<4x4x4xf32>) -> tensor<4x4x4xf32>
  %ctx_i = tensor.empty() : tensor<4x4x16xf32>
  %ctx_z = linalg.fill ins(%c0 : f32) outs(%ctx_i : tensor<4x4x16xf32>) -> tensor<4x4x16xf32>
  %ctx_b = linalg.batch_matmul ins(%attn, %vb : tensor<4x4x4xf32>, tensor<4x4x16xf32>) outs(%ctx_z : tensor<4x4x16xf32>) -> tensor<4x4x16xf32>
  // [NH,S,D] -> [S,NH,D] -> [S,H]
  %ctx_t_i = tensor.empty() : tensor<4x4x16xf32>
  %ctx_t = linalg.transpose ins(%ctx_b : tensor<4x4x16xf32>) outs(%ctx_t_i : tensor<4x4x16xf32>) permutation = [1, 0, 2]
  %ctx = tensor.collapse_shape %ctx_t [[0], [1, 2]] : tensor<4x4x16xf32> into tensor<4x64xf32>
  %o = func.call @linear64(%ctx, %wo) : (tensor<4x64xf32>, tensor<64x64xf32>) -> tensor<4x64xf32>
  %h2 = arith.addf %h_in, %o : tensor<4x64xf32>

  %fn = func.call @rms_norm(%h2, %ffn_nw) : (tensor<4x64xf32>, tensor<64xf32>) -> tensor<4x64xf32>
  %gate = func.call @linear_ffn_up(%fn, %wgate) : (tensor<4x64xf32>, tensor<128x64xf32>) -> tensor<4x128xf32>
  %up = func.call @linear_ffn_up(%fn, %wup) : (tensor<4x64xf32>, tensor<128x64xf32>) -> tensor<4x128xf32>
  %silu = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
    ins(%gate : tensor<4x128xf32>) outs(%gate : tensor<4x128xf32>) {
    ^bb0(%a: f32, %b: f32):
      %n = arith.negf %a : f32
      %e = math.exp %n : f32
      %one = arith.constant 1.0 : f32
      %d = arith.addf %one, %e : f32
      %s = arith.divf %a, %d : f32
      linalg.yield %s : f32
  } -> tensor<4x128xf32>
  %ff = arith.mulf %silu, %up : tensor<4x128xf32>
  %down = func.call @linear_ffn_down(%ff, %wdown) : (tensor<4x128xf32>, tensor<64x128xf32>) -> tensor<4x64xf32>
  %h3 = arith.addf %h2, %down : tensor<4x64xf32>

  %last = tensor.extract_slice %h3[3, 0] [1, 64] [1, 1] : tensor<4x64xf32> to tensor<1x64xf32>
  // rmsnorm 1x64 via broadcast trick: expand weight path using private helper on tiled
  %last4_i = tensor.empty() : tensor<4x64xf32>
  %last4 = tensor.insert_slice %last into %last4_i[0, 0] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<4x64xf32>
  %ln = func.call @rms_norm(%last4, %out_nw) : (tensor<4x64xf32>, tensor<64xf32>) -> tensor<4x64xf32>
  %ln1 = tensor.extract_slice %ln[0, 0] [1, 64] [1, 1] : tensor<4x64xf32> to tensor<1x64xf32>

  %wt_i = tensor.empty() : tensor<64x32xf32>
  %wt = linalg.transpose ins(%wout : tensor<32x64xf32>) outs(%wt_i : tensor<64x32xf32>) permutation = [1, 0]
  %yi = tensor.empty() : tensor<1x32xf32>
  %yz = linalg.fill ins(%c0 : f32) outs(%yi : tensor<1x32xf32>) -> tensor<1x32xf32>
  %y = linalg.matmul ins(%ln1, %wt : tensor<1x64xf32>, tensor<64x32xf32>) outs(%yz : tensor<1x32xf32>) -> tensor<1x32xf32>
  %logits = tensor.collapse_shape %y [[0, 1]] : tensor<1x32xf32> into tensor<32xf32>
  return %logits : tensor<32xf32>
}

func.func @decode(%token: tensor<i64>) -> tensor<32xf32> {
  %pad = arith.constant dense<0> : tensor<4xi64>
  %tok = tensor.extract %token[] : tensor<i64>
  %c3 = arith.constant 3 : index
  %tokens = tensor.insert %tok into %pad[%c3] : tensor<4xi64>
  %logits = func.call @prefill(%tokens) : (tensor<4xi64>) -> tensor<32xf32>
  return %logits : tensor<32xf32>
}

func.func @add(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.addf %a, %b : tensor<4xf32>
  return %0 : tensor<4xf32>
}
