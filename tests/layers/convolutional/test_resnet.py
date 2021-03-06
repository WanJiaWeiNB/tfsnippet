import numpy as np
import tensorflow as tf
from mock import mock, Mock

from tfsnippet.layers import *
from tfsnippet.utils import get_static_shape


class ResNetBlockTestCase(tf.test.TestCase):

    def test_resnet_general_block(self):
        class TestHelper(object):
            """A pseudo convolution function to test resnet block."""
            def __init__(self,
                         test_obj,
                         resize_at_exit,
                         in_channels,
                         out_channels,
                         kernel_size,
                         strides,
                         shortcut_kernel_size,
                         shortcut_force_conv,
                         activation_fn,
                         normalizer_fn,
                         dropout_fn,
                         after_conv_0,
                         after_conv_1):
                self.test_obj = test_obj
                self.resize_at_exit = resize_at_exit
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.kernel_size = kernel_size
                self.strides = strides
                self.shortcut_kernel_size = shortcut_kernel_size
                self.shortcut_force_conv = shortcut_force_conv
                self.activation_fn = activation_fn
                self.normalizer_fn = normalizer_fn
                self.dropout_fn = dropout_fn
                self.after_conv_0 = after_conv_0
                self.after_conv_1 = after_conv_1
                self.apply_shortcut = (
                        (strides != 1 and strides != (1, 1)) or
                        in_channels != out_channels or
                        shortcut_force_conv
                )

                def apply_fn(fn, x):
                    if fn is not None:
                        return fn(x)
                    return x

                self.shortcut_in = 'input'
                if self.apply_shortcut:
                    self.shortcut_out = 'shortcut({})'.format(self.shortcut_in)
                else:
                    self.shortcut_out = self.shortcut_in

                self.conv0_in = apply_fn(activation_fn,
                                         apply_fn(normalizer_fn, 'input'))
                self.conv0_out = 'conv({})'.format(self.conv0_in)
                self.conv0_out = apply_fn(self.after_conv_0, self.conv0_out)
                self.conv1_in = apply_fn(activation_fn,
                                         apply_fn(normalizer_fn,
                                                  apply_fn(dropout_fn,
                                                           self.conv0_out)))
                self.conv1_out = 'conv_1({})'.format(self.conv1_in)
                self.conv1_out = apply_fn(self.after_conv_1, self.conv1_out)
                self.final_out = self.shortcut_out + self.conv1_out

            def doTest(self):
                with mock.patch('tensorflow.convert_to_tensor', (lambda x: x)):
                    def shortcut_conv(*args, **kwargs):
                        return self(*args, **kwargs)

                    shortcut_conv = Mock(wraps=shortcut_conv)
                    output = resnet_general_block(
                        conv_fn=self,
                        input='input',
                        in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=self.kernel_size,
                        strides=self.strides,
                        shortcut_kernel_size=self.shortcut_kernel_size,
                        shortcut_force_conv=self.shortcut_force_conv,
                        shortcut_conv_fn=shortcut_conv,
                        resize_at_exit=self.resize_at_exit,
                        activation_fn=self.activation_fn,
                        normalizer_fn=self.normalizer_fn,
                        dropout_fn=self.dropout_fn
                    )
                    self.assertEqual(shortcut_conv.called, self.apply_shortcut)
                    self.assertEqual(output, self.final_out)

            def assertEqual(self, *args, **kwargs):
                return self.test_obj.assertEqual(*args, **kwargs)

            def __call__(self, input, out_channels, kernel_size, strides,
                         scope):
                if scope == 'shortcut':
                    if not self.apply_shortcut:
                        raise RuntimeError('shortcut should not be called.')
                    self.assertEqual(input, self.shortcut_in)
                    self.assertEqual(out_channels, self.out_channels)
                    self.assertEqual(kernel_size, self.shortcut_kernel_size)
                    self.assertEqual(strides, self.strides)
                    return self.shortcut_out
                elif scope == 'conv_0':
                    self.assertEqual(input, self.conv0_in)
                    if self.resize_at_exit:
                        self.assertEqual(out_channels, self.in_channels)
                        self.assertEqual(kernel_size, self.kernel_size)
                        self.assertEqual(strides, 1)
                    else:
                        self.assertEqual(out_channels, self.out_channels)
                        self.assertEqual(kernel_size, self.kernel_size)
                        self.assertEqual(strides, self.strides)
                    return self.conv0_out
                elif scope == 'conv_1':
                    self.assertEqual(input, self.conv1_in)
                    self.assertEqual(out_channels, self.out_channels)
                    self.assertEqual(kernel_size, self.kernel_size)
                    if self.resize_at_exit:
                        self.assertEqual(strides, self.strides)
                    else:
                        self.assertEqual(strides, 1)
                    return self.conv1_out
                else:
                    raise ValueError('Unexpected name: {!r}'.format(scope))

        def normalizer_fn(t):
            return 'normalizer_fn({})'.format(t)

        def activation_fn(t):
            return 'activation_fn({})'.format(t)

        def dropout_fn(t):
            return 'dropout_fn({})'.format(t)

        def after_conv_0(t):
            return 'after_conv_0({})'.format(t)

        def after_conv_1(t):
            return 'after_conv_1({})'.format(t)

        # test direct shortcut, resize at exit, w/o norm, act, dropout
        TestHelper(
            self,
            resize_at_exit=True,
            in_channels=5,
            out_channels=5,
            kernel_size=3,
            strides=1,
            shortcut_kernel_size=1,
            shortcut_force_conv=False,
            activation_fn=None,
            normalizer_fn=None,
            dropout_fn=None,
            after_conv_0=None,
            after_conv_1=None).doTest()
        TestHelper(
            self,
            resize_at_exit=False,
            in_channels=5,
            out_channels=5,
            kernel_size=3,
            strides=(1, 1),
            shortcut_kernel_size=1,
            shortcut_force_conv=False,
            activation_fn=None,
            normalizer_fn=None,
            dropout_fn=None,
            after_conv_0=None,
            after_conv_1=None).doTest()

        # test conv shortcut because of force conv, w/o norm, act, dropout
        TestHelper(
            self,
            resize_at_exit=True,
            in_channels=5,
            out_channels=5,
            kernel_size=3,
            strides=1,
            shortcut_kernel_size=1,
            shortcut_force_conv=True,
            activation_fn=None,
            normalizer_fn=None,
            dropout_fn=None,
            after_conv_0=None,
            after_conv_1=None).doTest()
        TestHelper(
            self,
            resize_at_exit=False,
            in_channels=5,
            out_channels=5,
            kernel_size=3,
            strides=(1, 1),
            shortcut_kernel_size=1,
            shortcut_force_conv=True,
            activation_fn=None,
            normalizer_fn=None,
            dropout_fn=None,
            after_conv_0=None,
            after_conv_1=None).doTest()

        # test conv shortcut because of channel mismatch, w norm, act, dropout
        TestHelper(
            self,
            resize_at_exit=True,
            in_channels=5,
            out_channels=7,
            kernel_size=3,
            strides=1,
            shortcut_kernel_size=11,
            shortcut_force_conv=False,
            activation_fn=normalizer_fn,
            normalizer_fn=activation_fn,
            dropout_fn=dropout_fn,
            after_conv_0=after_conv_0,
            after_conv_1=after_conv_1).doTest()
        TestHelper(
            self,
            resize_at_exit=False,
            in_channels=5,
            out_channels=7,
            kernel_size=3,
            strides=(1, 1),
            shortcut_kernel_size=11,
            shortcut_force_conv=False,
            activation_fn=normalizer_fn,
            normalizer_fn=activation_fn,
            dropout_fn=dropout_fn,
            after_conv_0=after_conv_0,
            after_conv_1=after_conv_1).doTest()

        # test conv shortcut because of strides > 1, w norm, act, dropout
        TestHelper(
            self,
            resize_at_exit=True,
            in_channels=5,
            out_channels=5,
            kernel_size=3,
            strides=2,
            shortcut_kernel_size=11,
            shortcut_force_conv=False,
            activation_fn=normalizer_fn,
            normalizer_fn=activation_fn,
            dropout_fn=dropout_fn,
            after_conv_0=after_conv_0,
            after_conv_1=after_conv_1).doTest()
        TestHelper(
            self,
            resize_at_exit=False,
            in_channels=5,
            out_channels=5,
            kernel_size=3,
            strides=(2, 1),
            shortcut_kernel_size=11,
            shortcut_force_conv=False,
            activation_fn=normalizer_fn,
            normalizer_fn=activation_fn,
            dropout_fn=dropout_fn,
            after_conv_0=after_conv_0,
            after_conv_1=after_conv_1).doTest()

    def test_resnet_conv2d_block(self):
        with mock.patch('tfsnippet.layers.convolutional.resnet.'
                        'resnet_general_block',
                        Mock(wraps=resnet_general_block)) as fn:
            normalizer_fn = lambda x: x
            activation_fn = lambda x: x
            dropout_fn = lambda x: x
            after_conv_0 = lambda x: x
            after_conv_1 = lambda x: x

            # test NHWC
            input = tf.constant(np.random.random(size=[17, 11, 32, 31, 5]),
                                dtype=tf.float32)
            output = resnet_conv2d_block(
                input=input,
                out_channels=7,
                kernel_size=3,
                name='conv_layer',
            )
            self.assertEqual(get_static_shape(output), (17, 11, 32, 31, 7))
            self.assertDictEqual(fn.call_args[1], {
                'input': input,
                'in_channels': 5,
                'out_channels': 7,
                'kernel_size': 3,
                'strides': (1, 1),
                'shortcut_kernel_size': (1, 1),
                'shortcut_force_conv': False,
                'resize_at_exit': True,
                'activation_fn': None,
                'normalizer_fn': None,
                'dropout_fn': None,
                'after_conv_0': None,
                'after_conv_1': None,
                'name': 'conv_layer',
                'scope': None,
            })

            # test NCHW
            input = tf.constant(np.random.random(size=[17, 11, 5, 32, 31]),
                                dtype=tf.float32)
            output = resnet_conv2d_block(
                input=input,
                out_channels=7,
                kernel_size=(3, 3),
                strides=2,
                channels_last=False,
                resize_at_exit=False,
                shortcut_force_conv=True,
                activation_fn=activation_fn,
                normalizer_fn=normalizer_fn,
                dropout_fn=dropout_fn,
                after_conv_0=after_conv_0,
                after_conv_1=after_conv_1,
                name='conv_layer',
            )
            self.assertEqual(get_static_shape(output), (17, 11, 7, 16, 16))
            self.assertDictEqual(fn.call_args[1], {
                'input': input,
                'in_channels': 5,
                'out_channels': 7,
                'kernel_size': (3, 3),
                'strides': 2,
                'shortcut_kernel_size': (1, 1),
                'shortcut_force_conv': True,
                'resize_at_exit': False,
                'activation_fn': activation_fn,
                'normalizer_fn': normalizer_fn,
                'dropout_fn': dropout_fn,
                'after_conv_0': after_conv_0,
                'after_conv_1': after_conv_1,
                'name': 'conv_layer',
                'scope': None,
            })

    def test_resnet_deconv2d_block(self):
        with mock.patch('tfsnippet.layers.convolutional.resnet.'
                        'resnet_general_block',
                        Mock(wraps=resnet_general_block)) as fn:
            normalizer_fn = lambda x: x
            activation_fn = lambda x: x
            dropout_fn = lambda x: x
            after_conv_0 = lambda x: x
            after_conv_1 = lambda x: x

            # test NHWC
            input = tf.constant(np.random.random(size=[17, 11, 32, 31, 5]),
                                dtype=tf.float32)
            output = resnet_deconv2d_block(
                input=input,
                out_channels=7,
                kernel_size=3,
                name='deconv_layer',
            )
            self.assertEqual(get_static_shape(output), (17, 11, 32, 31, 7))
            self.assertDictEqual(fn.call_args[1], {
                'input': input,
                'in_channels': 5,
                'out_channels': 7,
                'kernel_size': 3,
                'strides': (1, 1),
                'shortcut_kernel_size': (1, 1),
                'shortcut_force_conv': False,
                'resize_at_exit': False,
                'activation_fn': None,
                'normalizer_fn': None,
                'dropout_fn': None,
                'after_conv_0': None,
                'after_conv_1': None,
                'name': 'deconv_layer',
                'scope': None,
            })

            # test NCHW
            input = tf.constant(np.random.random(size=[17, 11, 5, 32, 31]),
                                dtype=tf.float32)
            output = resnet_deconv2d_block(
                input=input,
                out_channels=7,
                kernel_size=(3, 3),
                strides=2,
                channels_last=False,
                shortcut_force_conv=True,
                resize_at_exit=True,
                activation_fn=activation_fn,
                normalizer_fn=normalizer_fn,
                dropout_fn=dropout_fn,
                after_conv_0=after_conv_0,
                after_conv_1=after_conv_1,
                name='deconv_layer',
            )
            self.assertEqual(get_static_shape(output), (17, 11, 7, 64, 62))
            self.assertDictEqual(fn.call_args[1], {
                'input': input,
                'in_channels': 5,
                'out_channels': 7,
                'kernel_size': (3, 3),
                'strides': 2,
                'shortcut_kernel_size': (1, 1),
                'shortcut_force_conv': True,
                'resize_at_exit': True,
                'activation_fn': activation_fn,
                'normalizer_fn': normalizer_fn,
                'dropout_fn': dropout_fn,
                'after_conv_0': after_conv_0,
                'after_conv_1': after_conv_1,
                'name': 'deconv_layer',
                'scope': None,
            })
