module @paged_kv_abi {
  // Typed tensor lists remain illegal for exported entrypoints.
  util.func public @sum_page_heads_tensor_list(
      %pages: !util.list<tensor<*xf32>>) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    %page_count = util.list.size %pages : !util.list<tensor<*xf32>>
    %sum = scf.for %page_index = %c0 to %page_count step %c1
        iter_args(%acc = %zero) -> (f32) {
      %page = util.list.get %pages[%page_index]
          : !util.list<tensor<*xf32>> -> tensor<16xf32>
      %head = tensor.extract %page[%c0] : tensor<16xf32>
      %next = arith.addf %acc, %head : f32
      scf.yield %next : f32
    }
    util.return %sum : f32
  }

  // !util.list<!hal.buffer_view> is the supported paged KV ABI (v6).
  util.func public @touch_pages(
      %pages: !util.list<!hal.buffer_view>) -> !util.list<!hal.buffer_view> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %one = arith.constant 1.0 : f32
    %page_count = util.list.size %pages : !util.list<!hal.buffer_view>
    scf.for %page_index = %c0 to %page_count step %c1 {
      %bv = util.list.get %pages[%page_index]
          : !util.list<!hal.buffer_view> -> !hal.buffer_view
      %page = hal.tensor.import %bv "page" : !hal.buffer_view -> tensor<16xf32>
      %updated = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]}
          ins(%page : tensor<16xf32>) outs(%page : tensor<16xf32>) {
        ^bb0(%in: f32, %out: f32):
          %r = arith.addf %in, %one : f32
          linalg.yield %r : f32
      } -> tensor<16xf32>
      %out_bv = hal.tensor.export %updated "page_out"
          : tensor<16xf32> -> !hal.buffer_view
      util.list.set %pages[%page_index], %out_bv
          : !hal.buffer_view -> !util.list<!hal.buffer_view>
    }
    util.return %pages : !util.list<!hal.buffer_view>
  }
}
