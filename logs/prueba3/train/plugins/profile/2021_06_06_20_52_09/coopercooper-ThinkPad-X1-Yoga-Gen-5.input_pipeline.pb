	
ףp=?@
ףp=?@!
ףp=?@	?Y????@?Y????@!?Y????@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$
ףp=?@? %?4??A?y?]??@Y??:M??*	S㥛?X@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?c?????!?w???A@)???o^??1??54?<@:Preprocessing2F
Iterator::Model?????Q??!?Q,??B@)3?<Fy??1????6@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?	?Y?>??!?#"?4@)???ӝ'??1???E?.@:Preprocessing2U
Iterator::Model::ParallelMapV2	4??yT??!fy|??,@)	4??yT??1fy|??,@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorԜ???z?!??C??V@)Ԝ???z?1??C??V@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipu=?u???!???|pO@)????z?1Y??=(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?*?)?t?!?M$	??@)?*?)?t?1?M$	??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 18.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?Y????@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	? %?4??? %?4??!? %?4??      ??!       "      ??!       *      ??!       2	?y?]??@?y?]??@!?y?]??@:      ??!       B      ??!       J	??:M????:M??!??:M??R      ??!       Z	??:M????:M??!??:M??JCPU_ONLYY?Y????@b 