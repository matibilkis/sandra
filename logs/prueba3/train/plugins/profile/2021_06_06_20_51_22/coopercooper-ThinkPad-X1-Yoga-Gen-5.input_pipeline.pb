	?d?@?d?@!?d?@	???86
@???86
@!???86
@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?d?@??a?????A?b?0????Yp???????*	w?&1\@2F
Iterator::Model?ަ?????!?z?݅'G@)?????ɢ?11~ȕ_@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??k????!?
???;@)??{?q??1????Y,6@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?p?iݖ?!?j????3@)R~R?ӑ?1D?X?a/@:Preprocessing2U
Iterator::Model::ParallelMapV2f??! ??!?&T?+@)f??! ??1?&T?+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??Wή?!/?}"z?J@)͐*?WY{?1?TZ?@?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorPqx??y?!n3????@)Pqx??y?1n3????@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicex{?%t?!:.?׎@)x{?%t?1:.?׎@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 19.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???86
@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??a???????a?????!??a?????      ??!       "      ??!       *      ??!       2	?b?0?????b?0????!?b?0????:      ??!       B      ??!       J	p???????p???????!p???????R      ??!       Z	p???????p???????!p???????JCPU_ONLYY???86
@b 