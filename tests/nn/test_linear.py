# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for linear module in MS-AMP."""

import io
import copy
import unittest
import torch

from msamp.common.dtype import Dtypes
from msamp.common.tensor import ScalingTensor, ScalingMeta
from msamp.nn import LinearReplacer
from tests.helper import decorator
from msamp.common.tensor import TypeCast


class LinearTestCase(unittest.TestCase):
    """Test functions in FP8LInear and LinearReplacer."""
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        torch.manual_seed(1000)

    def tearDown(self) -> None:
        """Hook method for deconstructing the test fixture after testing it."""
        pass

    @decorator.cuda_test
    def test_fp8linear_forward(self):
        """Test FP8LInear forward function."""
        input = torch.randn((4, 4), device='cuda')
        linear = torch.nn.Linear(4, 8).cuda()
        for qtype in [Dtypes.kfloat32, Dtypes.kfloat16, Dtypes.kbfloat16]:
            model = LinearReplacer.replace(linear, qtype)

            output = linear(input)
            fp8_output = model(input)
            self.assertTrue(fp8_output.dtype == torch.float32)
            self.assertTrue(fp8_output.size() == torch.Size((4, 8)))
            self.assertTrue(torch.allclose(output, fp8_output, 0, 0.1))

    @decorator.cuda_test
    def test_fp8linear_backward(self):
        """Test FP8Linear backward function."""
        input = torch.randn((4, 4), device='cuda')
        linear = torch.nn.Linear(4, 8).cuda()
        linear_copy = copy.deepcopy(linear)

        linear(input).sum().backward()

        for qtype in [Dtypes.kfloat32, Dtypes.kfloat16, Dtypes.kbfloat16]:
            fp8linear = LinearReplacer.replace(linear_copy, qtype)
            fp8linear(input).sum().backward()

            # check bias.
            self.assertTrue(isinstance(fp8linear.bias.grad, torch.Tensor))
            self.assertTrue(torch.equal(fp8linear.bias.grad, linear.bias.grad))

            # check weight.
            self.assertTrue(isinstance(fp8linear.weight.grad, ScalingTensor))
            self.assertTrue(fp8linear.weight.grad.size() == linear.weight.grad.size())

    @decorator.cuda_test
    def test_fp8linear_accu_grad(self):
        """Test accumulate gradient in FP8Linear."""
        input = torch.randn((4, 4), device='cuda')
        linear = torch.nn.Linear(4, 4).cuda()

        model1 = copy.deepcopy(linear)
        model1 = LinearReplacer.replace(model1, Dtypes.kfloat16)
        output1 = model1(input)
        output1.sum().backward()

        model2 = copy.deepcopy(linear)
        model2 = LinearReplacer.replace(model2, Dtypes.kfloat16)
        for i in range(len(input)):
            input2 = input[i:i + 1]
            output2 = model2(input2)
            output2.sum().backward()
        self.assertTrue(torch.allclose(model1.weight.grad.float(), model2.weight.grad.float(), 0, 0.1))

    @decorator.cuda_test
    def test_fp8linear_parameters(self):
        """Test model's parameters of FP8Linear."""
        linear = torch.nn.Linear(4, 8).cuda()
        model = LinearReplacer.replace(linear, Dtypes.kfloat16)
        parameters = dict()
        for name, param in model.named_parameters():
            parameters[name] = param

        self.assertEqual(parameters['weight']._param_name, 'weight')
        self.assertTrue('weight' in parameters)
        self.assertTrue('bias' in parameters)
        self.assertTrue(isinstance(model.weight, ScalingTensor))
        self.assertTrue(torch.allclose(model.weight.float(), linear.weight, rtol=2e-4, atol=1e-3))
        self.assertTrue((linear.bias == model.bias).all())

    @decorator.cuda_test
    def test_meta_mem_immutable(self):
        """Test if meta memory is immutable in FP8Linear."""
        input = torch.randn(4, 4, device='cuda')
        linear = torch.nn.Linear(4, 4).cuda()
        model = LinearReplacer.replace(linear, Dtypes.kfloat16)
        model(input)
        # change mem
        amax = model.scaling_metas['input'].amax
        amax.data = amax.new_zeros(amax.shape)
        with self.assertRaises(RuntimeError):
            model(input)

    @decorator.cuda_test
    def test_linear_output_dtype(self):
        """Test output dtype of FP8Linear."""
        input = torch.randn(4, 4, device='cuda')
        linear = torch.nn.Linear(4, 8).cuda()
        model = LinearReplacer.replace(linear, Dtypes.kfloat16)
        self.assertEqual(model(input).dtype, torch.float32)
        with torch.cuda.amp.autocast():
            self.assertEqual(model(input).dtype, torch.float16)
            self.assertEqual(model(input.half()).dtype, torch.float16)
        model.half()
        self.assertEqual(model(input.half()).dtype, torch.float16)

    @decorator.cuda_test
    def test_linear_custom_attrs(self):
        """Test custom attrs of FP8Linear."""
        linear = torch.nn.Linear(4, 8).cuda()
        linear_attr_abc = 123
        weight_attr_abc = 42
        bias_attr_abc = 100
        linear.abc = linear_attr_abc
        linear.weight.abc = weight_attr_abc
        linear.bias.abc = bias_attr_abc
        model = LinearReplacer.replace(linear, Dtypes.kfloat16)
        # model
        self.assertFalse(model is linear)
        self.assertTrue(hasattr(model, 'abc'))
        self.assertEqual(model.abc, linear_attr_abc)
        # model.weight
        self.assertTrue(hasattr(model.weight, 'abc'))
        self.assertEqual(model.weight.abc, weight_attr_abc)
        # model.bias
        self.assertTrue(hasattr(model.bias, 'abc'))
        self.assertEqual(model.bias.abc, bias_attr_abc)

    @decorator.cuda_test
    def test_state_dict(self):
        """Test state dict of FP8Linear."""
        input = torch.randn((4, 4), device='cuda')
        linear = torch.nn.Linear(4, 8).cuda()
        model1 = LinearReplacer.replace(linear, Dtypes.kfloat16)

        state_dict = model1.state_dict()
        stream = io.BytesIO()
        torch.save(state_dict, stream)
        stream.seek(0)

        model2 = LinearReplacer.replace(linear, Dtypes.kfloat16)
        state_dict = torch.load(stream)
        model2.load_state_dict(state_dict)
        output1 = model1(input)
        output2 = model2(input)
        self.assertTrue(torch.equal(output1, output2))
        
    @decorator.cuda_test
    def test_activation_fp8(self):
        """Test FP8 activation in FP8Linear."""
        input = torch.randn((4, 4), device='cuda')
        linear = torch.nn.Linear(4, 8, bias=False).cuda()
        
        # for standard comparison
        model1 = copy.deepcopy(linear)
        model1 = LinearReplacer.replace(model1, Dtypes.kfloat16)
        output1 = model1(input)
        print(f"fp16 model output: {output1}, with dtype: {output1.dtype}, requires_grad: {output1.requires_grad}")
        
        print("------------For fp8 activation model------------")
        model2 = copy.deepcopy(linear)   
        model2 = LinearReplacer.replace(model2, Dtypes.kfloat16, enabling_fp8_activation=True)
        # input = input.cast(Dtypes.kfloat8_e4m3, meta=model2.scaling_metas['input'])       # 最新实现中（5.13）禁用了ScalingTensor的输入
        # print(hasattr(input, 'requires_grad'))        # True
        # print(input.requires_grad)                    # False
        # input.requires_grad = True            #! previously naive debug
        print(f"input: {input}, with dtype: {input.dtype}")
        
        output = model2(input)
        assert output.is_fp8_form == True
        print(f"output: {output}, with dtype: {output.dtype}, requires_grad: {output.requires_grad}, scaling_meta: {output.scaling_meta}")
        output = TypeCast.cast_from_fp8(output.view(dtype=torch.uint8), output.scaling_meta, Dtypes.kfloat32)
        print(f"casted output: {output}")
        print(f"difference: {output1 - output}")      
        
        # python -m unittest tests.nn.test_linear.LinearTestCase.test_activation_fp8
        
    @decorator.cuda_test
    def test_activation_fp8_multilayer(self):
        """Test FP8 activation in multi-layer FP8Linear."""
        input = torch.randn((4, 4), device='cuda')
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8, bias=False).cuda(),
            torch.nn.Linear(8, 8, bias=False).cuda(),
            torch.nn.Linear(8, 4, bias=False).cuda()
        )
        
        model1 = copy.deepcopy(model)
        model1 = LinearReplacer.replace(model1, Dtypes.kfloat16)
        output1 = model1(input)
        print(f"fp16 model output: {output1}, with dtype: {output1.dtype}")
        
        print("------------For fp8 activation model------------")
        model2 = copy.deepcopy(model)
        model2 = LinearReplacer.replace(model2, Dtypes.kfloat16, enabling_fp8_activation=True)
        
        print(model2)
        
        output = model2(input)
        assert output.is_fp8_form == True
        print(f"output: {output}, with dtype: {output.dtype}, requires_grad: {output.requires_grad}, scaling_meta: {output.scaling_meta}")
        # out_int8 = output.view(dtype=torch.uint8)       # fp16先view回int8，确保数值计算是对的
        # print("test if out tensor maintain scaling_meta after view: ", out_int8.scaling_meta)     #! None, indicates that the scaling_meta is not maintained after view
        output = TypeCast.cast_from_fp8(output.view(dtype=torch.uint8), output.scaling_meta, Dtypes.kfloat32)
        print(f"casted output: {output}")
        
        print(f"difference: {output1 - output}")
        
        # python -m unittest tests.nn.test_linear.LinearTestCase.test_activation_fp8_multilayer
    
    @decorator.cuda_test
    def test_activation_fp8_backward(self):
        """Test backward of FP8 activation in FP8Linear."""
        input = torch.randn((3, 4), device='cuda')
        linear = torch.nn.Linear(4, 8, bias=False).cuda()
        
        # for standard comparison
        model1 = copy.deepcopy(linear)
        model1 = LinearReplacer.replace(model1, Dtypes.kfloat16)
        print(f"Input for fp16 requires_grad: {input.requires_grad}")
        output1 = model1(input)
        print(f"fp16 model output: {output1}, with dtype: {output1.dtype}, with requires_grad: {output1.requires_grad}, with grad_fn: {output1.grad_fn}")
        loss = output1.sum()
        print(f">>> output1.sum: {loss}, with requires_grad: {loss.requires_grad}, with grad_fn: {loss.grad_fn}")
        loss.backward()
        #! with dtype: torch.float32, with requires_grad: True, with grad_fn: <AddBackward0 object at 0x7f32806ceb90>
        print(f"fp16 model weight grad: {model1.weight.grad}, with dtype: {model1.weight.grad.dtype}, with requires_grad: {model1.weight.grad._requires_grad}")
        
        print("------------For fp8 activation model------------")
        model2 = copy.deepcopy(linear)   
        model2 = LinearReplacer.replace(model2, Dtypes.kfloat16, enabling_fp8_activation=True)
        # input = input.cast(Dtypes.kfloat8_e4m3, meta=model2.scaling_metas['input'])
        # print(f"input: {input}, with dtype: {input.dtype}, with requires_grad: {input._requires_grad}")
        
        output2 = model2(input)
        assert output2.is_fp8_form == True
        print(f"FP8 model output: {output2}, with dtype: {output2.dtype}, with requires_grad: {output2.requires_grad}, with grad_fn: {output2.grad_fn}")
        
        #! we must make sure that the loss function contains the gradient
        class ScalingSum(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                meta = input.scaling_meta
                ctx.shape = input.shape     # torch.Size([3,2]), 这里是view前FP16 tensor的形状，为的是backward返回到形状不会出错（要和input的value形状保持一致）
                input = input.view(dtype=torch.uint8)
                ctx.true_out_shape = input.shape    # torch.Size([3,4]), 这里是view后FP8 tensor的形状
                input = TypeCast.cast_from_fp8(input, meta, Dtypes.kfloat32)
                return input.sum()
            
            @staticmethod
            def backward(ctx, grad_output):
                activation_grad = grad_output * torch.ones(ctx.true_out_shape, dtype=torch.float16).cuda()  # torch.Size([3,4])
                
                activation_grad_scaling_tensor = activation_grad.cast(Dtypes.kfloat8_e5m2, meta=ScalingMeta(Dtypes.kfloat8_e5m2))    # todo: 在这个SclingSum函数里面，只是为了传回去的activation_grad有scaling_meta这个属性，但是这里这个meta是没有用的，因为毕竟Sum函数的grad_fn是全一。如果之后要实现别的自定义损失函数，这里的meta就有用了，而且可能要考虑forward中的meta值
                activation_grad = activation_grad_scaling_tensor.value.view(dtype=torch.float16)      # torch.Size([3,4]) to torch.Size([3,2])
                activation_grad.scaling_meta = activation_grad_scaling_tensor.meta
                activation_grad.is_fp8_form = True
                
                assert activation_grad.shape == ctx.shape, f"Activation grad shape should be the same as input shape {ctx.shape}, but got {activation_grad.shape}"
                print(f"ScalingSum return: {activation_grad}")
                return activation_grad
            
        loss = ScalingSum.apply(output2)
        print(f">>> output2.sum: {loss}, with requires_grad: {loss.requires_grad}, with grad_fn: {loss.grad_fn}")
        loss.backward()
        
        print(f"weight grad: {model2.weight.grad}, with dtype: {model1.weight.grad.dtype}, with requires_grad: {model1.weight.grad._requires_grad}")     #! None
        #! 5.10 此时weight grad shape为4*4, weight shape为4*8，这是因为Gemm func里面还未处理view导致tensor的张量形状变化问题，uint8的张量view成float16的会导致tensor最后一维大小÷2 (solved 5.13)
        assert model2.weight.shape == model2.weight.grad.shape, f"Weight grad shape should be the same as model weight shape {model2.weight.shape}, but got {model2.weight.grad.shape}"
        
        # python -m unittest tests.nn.test_linear.LinearTestCase.test_activation_fp8_backward
        
    @decorator.cuda_test
    def test_activation_fp8_backward_multilayer(self):
        """Test backward of FP8 activation in multi-layer FP8Linear."""
        input = torch.randn((3, 4), device='cuda')
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8, bias=False).cuda(),
            torch.nn.Linear(8, 8, bias=False).cuda(),
            torch.nn.Linear(8, 4, bias=False).cuda()
        )
        
        model1 = copy.deepcopy(model)
        model1 = LinearReplacer.replace(model1, Dtypes.kfloat16)
        output1 = model1(input)
        print(f"fp16 model output: {output1}, with dtype: {output1.dtype}, with requires_grad: {output1.requires_grad}, with grad_fn: {output1.grad_fn}")
        loss = output1.sum()
        print(f"output1.sum: {loss}, with requires_grad: {loss.requires_grad}, with grad_fn: {loss.grad_fn}")
        loss.backward()
        print(f"fp16 model weight grad: {model1[0].weight.grad}, with dtype: {model1[0].weight.grad.dtype}, with requires_grad: {model1[0].weight.grad._requires_grad}")
        
        print("------------For fp8 activation model------------")
        model2 = copy.deepcopy(model)   
        model2 = LinearReplacer.replace(model2, Dtypes.kfloat16, enabling_fp8_activation=True)
        output2 = model2(input)
        assert output2.is_fp8_form == True
        print(f"FP8 model output: {output2}, with dtype: {output2.dtype}, with requires_grad: {output2.requires_grad}, with grad_fn: {output2.grad_fn}")
        
        class ScalingSum(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                meta = input.scaling_meta
                ctx.shape = input.shape     # torch.Size([3,2]), 这里是view前FP16 tensor的形状，为的是backward返回到形状不会出错（要和input的value形状保持一致）
                input = input.view(dtype=torch.uint8)
                ctx.true_out_shape = input.shape    # torch.Size([3,4]), 这里是view后FP8 tensor的形状
                input = TypeCast.cast_from_fp8(input, meta, Dtypes.kfloat32)
                return input.sum()
            
            @staticmethod
            def backward(ctx, grad_output):
                activation_grad = grad_output * torch.ones(ctx.true_out_shape, dtype=torch.float16).cuda()  # torch.Size([3,4])
                
                activation_grad_scaling_tensor = activation_grad.cast(Dtypes.kfloat8_e5m2, meta=ScalingMeta(Dtypes.kfloat8_e5m2))    # todo: 在这个SclingSum函数里面，只是为了传回去的activation_grad有scaling_meta这个属性，但是这里这个meta是没有用的，因为毕竟Sum函数的grad_fn是全一。如果之后要实现别的自定义损失函数，这里的meta就有用了，而且可能要考虑forward中的meta值
                activation_grad = activation_grad_scaling_tensor.value.view(dtype=torch.float16)      # torch.Size([3,4]) to torch.Size([3,2])
                activation_grad.scaling_meta = activation_grad_scaling_tensor.meta
                activation_grad.is_fp8_form = True
                
                assert activation_grad.shape == ctx.shape, f"Activation grad shape should be the same as input shape {ctx.shape}, but got {activation_grad.shape}"
                print(f"ScalingSum return: {activation_grad}")
                return activation_grad
            
        loss = ScalingSum.apply(output2)
        print(f"loss (ScalingSum): {loss}, with requires_grad: {loss.requires_grad}, with grad_fn: {loss.grad_fn}")
        # loss = output2.sum()    #! 这里暂时还不能直接sum，因为output2是view后的fp16 tensor，数值是不对的，要自定义一个能正确处理viewed tensor+meta输入的损失函数
        # print(f"output2.sum: {loss}, with requires_grad: {loss.requires_grad}, with grad_fn: {loss.grad_fn}")
        
        loss.backward()
        print(f"fp8 model weight grad: {model2[0].weight.grad}, with dtype: {model2[0].weight.grad.dtype}, with requires_grad: {model2[0].weight.grad._requires_grad}")
        
        # python -m unittest tests.nn.test_linear.LinearTestCase.test_activation_fp8_backward_multilayer
