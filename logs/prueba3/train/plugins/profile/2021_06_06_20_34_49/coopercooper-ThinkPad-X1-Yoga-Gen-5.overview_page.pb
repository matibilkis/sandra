?	?|гY????|гY???!?|гY???	k_3???	@k_3???	@!k_3???	@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?|гY?????P?\???AOI?V??Y?խ??ޯ?*	@5^?I?U@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??kC?8??!K??5oA@)v?Kp???1?r???;@:Preprocessing2U
Iterator::Model::ParallelMapV2?g@?5??!?2)m1@)?g@?5??1?2)m1@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?;?2TŔ?!??I?t27@)e?pu č?1?@??0@:Preprocessing2F
Iterator::Modeld???????!?gb?>?@@)????G??1Ғ%??b/@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip+1?JZ???!??κ`?P@)?[?~l??1s?E?!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?4Lk?x?!Z?yι@)?4Lk?x?1Z?yι@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceZd;?O?w?!e?4?M@)Zd;?O?w?1e?4?M@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9k_3???	@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??P?\?????P?\???!??P?\???      ??!       "      ??!       *      ??!       2	OI?V??OI?V??!OI?V??:      ??!       B      ??!       J	?խ??ޯ??խ??ޯ?!?խ??ޯ?R      ??!       Z	?խ??ޯ??խ??ޯ?!?խ??ޯ?JCPU_ONLYYk_3???	@b Y      Y@q.Pf?n@"?
both?Your program is POTENTIALLY input-bound because 4.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 