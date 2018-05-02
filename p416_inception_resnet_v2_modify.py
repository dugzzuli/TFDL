import os
import tensorflow as tf
from datasets import flowers
import inception_resnet_v2_a as model
#import inception_preprocessing
import tensorflow.contrib.slim as slim


checkpoint_dir = "./data/model"
image_size = 299

def get_init_fn(sess):
    # 不进行载入的层次名称
    checkpoint_exclude_scopes = ["InceptionResnetV2/AuxLogits", "InceptionResnetV2/Logits"]
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    # 记录需要载入的参数
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    # 重新设置 init-fn 函数，参数列表使用已去除定义层次的列表
    model_path = os.path.join(checkpoint_dir, "inception_resnet_v2_2016_08_30.ckpt")
    init_fn = slim.assign_from_checkpoint_fn(model_path, variables_to_restore)

    return init_fn(sess)


with tf.Graph().as_default():
    img = tf.random_normal([1, 299, 299, 3])

    with slim.arg_scope(model.inception_resnet_v2_arg_scope()):
        pre, _ = model.inception_resnet_v2(img, is_training=False)

    with tf.Session() as sess:

        init_fn = get_init_fn(sess)
        res = (sess.run(pre))
        print(res.shape)

