module @paged_kv_abi {
  util.func public @sum_page_heads(
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
}
