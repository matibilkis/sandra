  *	f??|[?@2T
Iterator::Prefetch::Generator???>?R@! j6Z?IV@)???>?R@1 j6Z?IV@:Preprocessing2I
Iterator::Prefetch?^?2????!?&??X@)?^?2????1?&??X@:Preprocessing2F
Iterator::Model?Ά?3???!???/oB@)??z?<d??1?z.?:9??:Preprocessing2?
TIterator::Model::Prefetch::Shard::Rebatch::Map::ParallelMapV2::Zip[1]::ForeverRepeat?ɋL????!TN?$?|??)N?#Ed??16,"??:Preprocessing2P
Iterator::Model::Prefetch	?%qVD??!!?"G???)	?%qVD??1!?"G???:Preprocessing2?
NIterator::Model::Prefetch::Shard::Rebatch::Map::ParallelMapV2::Zip[0]::FlatMapX?eS???!?ܕ??o??)LnYk(??1r??r??:Preprocessing2e
.Iterator::Model::Prefetch::Shard::Rebatch::Map<ۣ7?G??!???p????)	??YK??1?qS????:Preprocessing2W
 Iterator::Model::Prefetch::ShardA?شR??!?Ђ??L??)L7?A`???1"je???:Preprocessing2t
=Iterator::Model::Prefetch::Shard::Rebatch::Map::ParallelMapV2Q??9???!??x?%??)Q??9???1??x?%??:Preprocessing2?
^Iterator::Model::Prefetch::Shard::Rebatch::Map::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSliceɬ??vh??!6?J"???)ɬ??vh??16?J"???:Preprocessing2`
)Iterator::Model::Prefetch::Shard::Rebatch??????!v??pqx??)????Ɓ?1-D?ᰔ??:Preprocessing2y
BIterator::Model::Prefetch::Shard::Rebatch::Map::ParallelMapV2::Zip?k?,	P??!?EY?/@)??D????1??ـ7???:Preprocessing2?
`Iterator::Model::Prefetch::Shard::Rebatch::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??T???d?!m???????)??T???d?1m???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.