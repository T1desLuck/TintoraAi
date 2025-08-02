import torch
from src.model.tintora_ai import (
    DoubleConv, AttentionBlock, UNet, Generator, Discriminator, TintoraAI
)


def test_model_forward():
    """Тест прямого прохода модели с различными размерами входных изображений"""
    print("Starting test_model_forward")
    model = TintoraAI(num_classes=100)

    # Тестируем разные размеры входных данных
    for size in [64, 128, 256]:
        print(f"Testing input size {size}x{size}")
        input_tensor = torch.randn(1, 1, size, size)

        try:
            color_output, semantic_output = model(input_tensor)

            # Проверяем размеры выходных тензоров
            assert color_output.shape[0] == 1, f"Batch size should be 1 for size {size}x{size}"
            assert color_output.shape[1] == 3, f"Color output should have 3 channels for size {size}x{size}"
            assert color_output.shape[2:] == input_tensor.shape[2:], "Color output shape mismatch"
            assert semantic_output.shape == (1, 100), "Semantic output shape mismatch"

            print(f"✓ Color output shape: {color_output.shape}")
            print(f"✓ Semantic output shape: {semantic_output.shape}")

        except Exception as e:
            print(f"✗ Error testing size {size}x{size}: {e}")
            raise

    print("✓ Test completed successfully")


def test_rectangle_images():
    """Тест обработки прямоугольных изображений"""
    print("Starting test_rectangle_images")
    model = TintoraAI(num_classes=100)

    # Тестируем прямоугольные размеры
    for size in [(64, 128), (256, 128), (512, 256)]:
        print(f"Testing input size {size[0]}x{size[1]}")
        input_tensor = torch.randn(1, 1, size[0], size[1])

        try:
            color_output, semantic_output = model(input_tensor)

            # Проверяем размеры выходных тензоров
            assert color_output.shape[0] == 1, "Batch size should be 1"
            assert color_output.shape[1] == 3, "Color output should have 3 channels"
            assert color_output.shape[2:] == input_tensor.shape[2:], "Color output shape mismatch"
            assert semantic_output.shape == (1, 100), "Semantic output shape mismatch"

            print(f"✓ Color output shape: {color_output.shape}")
            print(f"✓ Semantic output shape: {semantic_output.shape}")

        except Exception as e:
            print(f"✗ Error testing size {size[0]}x{size[1]}: {e}")
            raise

    print("✓ Rectangle test completed successfully")


def test_double_conv():
    """Тест блока двойной свертки"""
    print("Starting test_double_conv")
    model = DoubleConv(in_channels=1, out_channels=64)
    input_tensor = torch.randn(1, 1, 64, 64)

    output = model(input_tensor)
    assert output.shape == (1, 64, 64, 64), "DoubleConv output shape mismatch"

    print("✓ DoubleConv test passed")


def test_attention_block():
    """Тест блока внимания"""
    print("Starting test_attention_block")
    model = AttentionBlock(channels=64)
    input_tensor = torch.randn(1, 64, 32, 32)

    output = model(input_tensor)
    assert output.shape == (1, 64, 32, 32), "AttentionBlock output shape mismatch"

    print("✓ AttentionBlock test passed")


def test_unet():
    """Тест архитектуры U-Net"""
    print("Starting test_unet")
    model = UNet(in_channels=1, out_channels=3)

    # Проверяем различные размеры входных данных
    for size in [64, 128]:
        input_tensor = torch.randn(1, 1, size, size)
        output = model(input_tensor)
        assert output.shape == (1, 3, size, size), f"UNet output shape mismatch for size {size}x{size}"

    print("✓ UNet test passed")


def test_generator():
    """Тест генератора для GAN"""
    print("Starting test_generator")
    model = Generator(in_channels=1, out_channels=3)
    input_tensor = torch.randn(1, 1, 64, 64)

    output = model(input_tensor)
    assert output.shape == (1, 3, 64, 64), "Generator output shape mismatch"

    # Проверяем значения (должны быть в пределах [0, 1])
    assert torch.all(output >= 0) and torch.all(output <= 1), "Generator output values outside [0, 1] range"

    print("✓ Generator test passed")


def test_discriminator():
    """Тест дискриминатора для GAN"""
    print("Starting test_discriminator")
    model = Discriminator(in_channels=4)  # 1 (ЧБ) + 3 (цветное) каналы
    img_bw = torch.randn(1, 1, 64, 64)
    img_color = torch.randn(1, 3, 64, 64)

    output = model(img_bw, img_color)
    # PatchGAN выход должен быть меньше из-за свертки
    assert output.shape[0] == 1, "Discriminator batch size mismatch"
    assert output.shape[1] == 1, "Discriminator output channels mismatch"

    print(f"✓ Discriminator output shape: {output.shape}")
    print("✓ Discriminator test passed")


def test_tintora_forward_mode():
    """Тест переключения режимов работы TintoraAI"""
    print("Starting test_tintora_forward_mode")
    model = TintoraAI(num_classes=100)
    input_tensor = torch.randn(1, 1, 64, 64)

    # Обычный режим (инференс)
    model.training_gan = False
    out1, sem1 = model(input_tensor)
    assert len(out1.shape) == 4, "Color output should be a 4D tensor"
    assert len(sem1.shape) == 2, "Semantic output should be a 2D tensor"

    # Режим обучения GAN
    model.training_gan = True
    out2, sem2, disc = model(input_tensor)
    assert len(out2.shape) == 4, "Color output should be a 4D tensor"
    assert len(sem2.shape) == 2, "Semantic output should be a 2D tensor"
    assert len(disc.shape) >= 3, "Discriminator output should be at least a 3D tensor"

    print("✓ TintoraAI forward mode test passed")


if __name__ == "__main__":
    test_model_forward()
    test_rectangle_images()
    test_double_conv()
    test_attention_block()
    test_unet()
    test_generator()
    test_discriminator()
    test_tintora_forward_mode()
    print("All tests passed!")
