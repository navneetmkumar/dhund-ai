from towhee import pipe, ops, DataCollection

img_pipe = (
    pipe.input('url')
    .map('url', 'img', ops.image_decode.cv2_rgb())
    .map('img', 'vec', ops.image_text_embedding.blip(model_name='blip_itm_base_coco', modality='image'))
    .output('img', 'vec')
)

text_pipe = (
    pipe.input('text')
    .map('text', 'vec', ops.image_text_embedding.blip(model_name='blip_itm_base_coco', modality='text'))
    .output('text', 'vec')
)
blip_op = towhee.ops.image_text_embedding.blip(model_name='blip_itm_base_coco', modality='image').get_op()

training_args = {
    'num_train_epochs': 3, # you can add epoch number to get a better metric.
    'per_device_train_batch_size': 8,
    'per_device_eval_batch_size': 8,
    'do_train': True,
    'do_eval': True,
    'remove_unused_columns': False,
    'output_dir': './tmp/test-blip',
    'overwrite_output_dir': True,
}
model_args = {
    'freeze_vision_model': False,
    'freeze_text_model': False,
    'cache_dir': './cache'
}

blip_op.train(data_args=data_args, training_args=training_args, model_args=model_args)