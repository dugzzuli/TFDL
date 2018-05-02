import tensorflow as tf
import p359_resnet_tool as tool


def res_unit(input_data, out_channels, i = 0):
    with tf.variable_scope("resUnit_", str(i)):
        output_data = tool.batch_norm(input_data)
        output_data = tool.relu(output_data, name="relu")
        output_data = tool.conv("conv1", output_data, out_channels[0], kernel_size=[1, 1])

        output_data = tool.batch_norm(output_data)
        output_data = tool.relu(output_data, name="relu")
        output_data = tool.conv("conv2", output_data, out_channels[1], kernel_size=[3, 3])

        output_data = tool.batch_norm(output_data)
        output_data = tool.relu(output_data, name="relu")
        output_data = tool.conv("conv3", output_data, out_channels[2], kernel_size=[1, 1])

        return output_data + input_data


def train(input_data):

    output_data = tool.batch_norm(input_data)
    output_data = tool.conv("first_layer", output_data, 32, [7, 7])
    output_data = tool.max_pool(output_data, 3, 3, 2, 2, name="maxpool1")

    output_data = res_unit(output_data, [16, 16, 32], 0)

