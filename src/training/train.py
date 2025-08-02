import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import os
import numpy as np
import argparse
from tqdm import tqdm
from pytorch_msssim import ssim
import yaml
import time
from torch.amp import autocast
from torch.amp.grad_scaler import GradScaler

from src.model.tintora_ai import TintoraAI
from src.training.dataset import ColorizationDataset
from src.training.gan_loss import GANLoss, PerceptualLoss


def train(config_path="config.yaml"):
    """Функция обучения модели TintoraAI с поддержкой GAN и различных размеров изображений"""
    # Загружаем конфигурацию
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Определяем устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # Инициализируем модель
    model = TintoraAI(num_classes=config['num_classes']).to(device)
    print("Модель инициализирована с нуля, без предобученных весов")

    # Настраиваем оптимизаторы
    optimizer_G = optim.Adam(model.generator.parameters(), lr=config['lr'])
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=config['lr'] * 0.1)
    optimizer_C = optim.Adam(model.classifier.parameters(), lr=config['lr'])

    # Настраиваем планировщики скорости обучения
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=5)
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=5)
    scheduler_C = optim.lr_scheduler.ReduceLROnPlateau(optimizer_C, mode='min', factor=0.5, patience=5)

    # Определяем функции потерь
    criterion_color = nn.L1Loss()  # L1 потеря лучше для цветовых значений
    criterion_class = nn.CrossEntropyLoss()
    criterion_gan = GANLoss(use_lsgan=True).to(device)
    criterion_perceptual = PerceptualLoss().to(device)

    # Подготавливаем датасеты
    dataset = ColorizationDataset(
        os.path.join(config['data_path'], "bw"),
        os.path.join(config['data_path'], "color"),
        os.path.join(config['data_path'], "labels"),
        augment=config.get('augment', True),
        min_size=config.get('min_image_size', 64),
        max_size=config.get('max_image_size', 1024)
    )

    # Выводим информацию о датасете
    print(f"Загружено {len(dataset)} изображений для обучения")

    # Разделяем на обучающую и валидационную выборки
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    print(f"Обучающий набор: {train_size} изображений")
    print(f"Валидационный набор: {val_size} изображений")

    # Создаём даталоадеры
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # Инициализируем GradScaler для смешанной точности
    scaler_G = GradScaler('cuda') if torch.cuda.is_available() else None
    scaler_D = GradScaler('cuda') if torch.cuda.is_available() else None
    scaler_C = GradScaler('cuda') if torch.cuda.is_available() else None

    # Параметры обучения
    epochs = config['epochs']
    accum_steps = config['accum_steps']
    lambda_perceptual = config.get('lambda_perceptual', 10.0)
    lambda_gan = config.get('lambda_gan', 1.0)
    save_path = config['save_path']

    # Создаём директорию для сохранения моделей
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Переменные для отслеживания лучшей модели
    best_val_loss = float('inf')
    patience = config.get('patience', 15)
    no_improve_count = 0

    # Цикл обучения
    for epoch in range(epochs):
        model.train()
        start_time = time.time()

        # Обучение на тренировочном наборе
        running_loss_color = 0.0
        running_loss_class = 0.0
        running_loss_gan_G = 0.0
        running_loss_gan_D = 0.0
        running_ssim = 0.0
        batch_count = 0

        # Переключаем модель в режим обучения GAN
        model.training_gan = True

        for i, (bw_images, color_images, labels) in enumerate(tqdm(train_loader, desc=f"Эпоха {epoch+1}/{epochs}")):
            bw_images = bw_images.to(device)
            color_images = color_images.to(device)
            labels = labels.to(device)

            # Обучаем дискриминатор
            # -----------------------
            # Очищаем градиенты
            optimizer_D.zero_grad()

            with autocast('cuda', enabled=scaler_D is not None):
                # Создаём реальные и поддельные входные данные для дискриминатора
                fake_output, _, _ = model(bw_images)

                # Рассчитываем потери для реальных и поддельных изображений
                d_real = model.discriminator(bw_images, color_images)
                d_fake = model.discriminator(bw_images, fake_output.detach())

                loss_d_real = criterion_gan(d_real, True)
                loss_d_fake = criterion_gan(d_fake, False)
                loss_D = (loss_d_real + loss_d_fake) / 2

            if scaler_D:
                scaler_D.scale(loss_D).backward()
                if (i + 1) % accum_steps == 0 or i == len(train_loader) - 1:
                    scaler_D.step(optimizer_D)
                    scaler_D.update()
            else:
                loss_D.backward()
                if (i + 1) % accum_steps == 0 or i == len(train_loader) - 1:
                    optimizer_D.step()

            # Обучаем генератор и классификатор
            # ---------------------------------
            optimizer_G.zero_grad()
            optimizer_C.zero_grad()

            with autocast('cuda', enabled=scaler_G is not None):
                # Вычисляем выходные данные модели
                fake_output, semantic_output, d_fake_gen = model(bw_images)

                # Рассчитываем потери
                loss_color = criterion_color(fake_output, color_images)
                loss_perceptual = criterion_perceptual(fake_output, color_images)

                # Преобразуем метки из формы [batch_size, 1] в [batch_size]
                loss_class = criterion_class(semantic_output, labels.squeeze())

                loss_gan_G = criterion_gan(d_fake_gen, True)

                # Комбинируем потери для генератора
                loss_G = loss_color + lambda_perceptual * loss_perceptual + lambda_gan * loss_gan_G

            # Обновляем генератор
            if scaler_G:
                scaler_G.scale(loss_G).backward(retain_graph=True)
                if (i + 1) % accum_steps == 0 or i == len(train_loader) - 1:
                    scaler_G.step(optimizer_G)
                    scaler_G.update()
            else:
                loss_G.backward(retain_graph=True)
                if (i + 1) % accum_steps == 0 or i == len(train_loader) - 1:
                    optimizer_G.step()

            # Обновляем классификатор отдельно
            with autocast('cuda', enabled=scaler_C is not None):
                loss_C = loss_class

            if scaler_C:
                scaler_C.scale(loss_C).backward()
                if (i + 1) % accum_steps == 0 or i == len(train_loader) - 1:
                    scaler_C.step(optimizer_C)
                    scaler_C.update()
            else:
                loss_C.backward()
                if (i + 1) % accum_steps == 0 or i == len(train_loader) - 1:
                    optimizer_C.step()

            # Рассчитываем SSIM для мониторинга качества
            with torch.no_grad():
                # Проверка наличия одинаковых размеров для SSIM
                if fake_output.shape[-2:] != color_images.shape[-2:]:
                    resized_fake = torch.nn.functional.interpolate(
                        fake_output, size=color_images.shape[-2:], mode='bilinear', align_corners=True)
                    current_ssim = ssim(resized_fake, color_images, data_range=1.0, size_average=True).item()
                else:
                    current_ssim = ssim(fake_output, color_images, data_range=1.0, size_average=True).item()

            # Накапливаем статистику
            running_loss_color += loss_color.item()
            running_loss_class += loss_class.item()
            running_loss_gan_G += loss_gan_G.item()
            running_loss_gan_D += loss_D.item()
            running_ssim += current_ssim
            batch_count += 1

            # Сохраняем примеры для визуального контроля (каждые 100 батчей)
            if i % 100 == 0 and config.get('save_samples', False):
                samples_dir = os.path.join(os.path.dirname(save_path), 'samples')
                os.makedirs(samples_dir, exist_ok=True)

                # Сохраняем первое изображение из батча
                with torch.no_grad():
                    # Преобразуем в изображения
                    bw_img = bw_images[0].cpu().squeeze().numpy() * 255
                    color_img = color_images[0].cpu().permute(1, 2, 0).numpy() * 255
                    fake_img = fake_output[0].cpu().permute(1, 2, 0).numpy() * 255

                    # Сохраняем
                    Image.fromarray(bw_img.astype(np.uint8)).save(
                        os.path.join(samples_dir, f'epoch{epoch+1}_batch{i}_bw.png'))
                    Image.fromarray(color_img.astype(np.uint8)).save(
                        os.path.join(samples_dir, f'epoch{epoch+1}_batch{i}_real.png'))
                    Image.fromarray(fake_img.astype(np.uint8)).save(
                        os.path.join(samples_dir, f'epoch{epoch+1}_batch{i}_fake.png'))

        # Вычисляем средние значения для эпохи
        avg_loss_color = running_loss_color / batch_count
        avg_loss_class = running_loss_class / batch_count
        avg_loss_gan_G = running_loss_gan_G / batch_count
        avg_loss_gan_D = running_loss_gan_D / batch_count
        avg_ssim = running_ssim / batch_count

        # Переключаем модель в режим валидации (без GAN)
        model.training_gan = False
        model.eval()

        # Валидация
        val_loss_color = 0.0
        val_loss_class = 0.0
        val_ssim = 0.0
        val_batch_count = 0

        with torch.no_grad():
            for bw_images, color_images, labels in val_loader:
                bw_images = bw_images.to(device)
                color_images = color_images.to(device)
                labels = labels.to(device)

                color_output, semantic_output = model(bw_images)

                # Проверка наличия одинаковых размеров
                if color_output.shape[-2:] != color_images.shape[-2:]:
                    color_output = torch.nn.functional.interpolate(
                        color_output, size=color_images.shape[-2:], mode='bilinear', align_corners=True)

                loss_color = criterion_color(color_output, color_images)

                # Преобразуем метки из формы [batch_size, 1] в [batch_size]
                loss_class = criterion_class(semantic_output, labels.squeeze())

                current_ssim = ssim(color_output, color_images, data_range=1.0, size_average=True).item()

                val_loss_color += loss_color.item()
                val_loss_class += loss_class.item()
                val_ssim += current_ssim
                val_batch_count += 1

        # Вычисляем средние значения для валидации
        avg_val_loss_color = val_loss_color / val_batch_count
        avg_val_loss_class = val_loss_class / val_batch_count
        avg_val_ssim = val_ssim / val_batch_count
        total_val_loss = avg_val_loss_color + avg_val_loss_class

        # Обновляем планировщики скорости обучения
        scheduler_G.step(total_val_loss)
        scheduler_D.step(avg_loss_gan_D)
        scheduler_C.step(avg_val_loss_class)

        # Проверяем, улучшилась ли модель
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            no_improve_count = 0

            # Сохраняем лучшую модель
            best_model_path = os.path.join(os.path.dirname(save_path), 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Сохранена лучшая модель с валидационной ошибкой: {total_val_loss:.4f}")
        else:
            no_improve_count += 1

        # Вычисляем время эпохи
        epoch_time = time.time() - start_time

        # Выводим статистику за эпоху
        print(f"Эпоха {epoch+1}/{epochs} (Время: {epoch_time:.1f}с)")
        print(f"Обучение - Color Loss: {avg_loss_color:.4f}, Class Loss: {avg_loss_class:.4f}, "
              f"GAN_G Loss: {avg_loss_gan_G:.4f}, GAN_D Loss: {avg_loss_gan_D:.4f}, SSIM: {avg_ssim:.4f}")
        print(f"Валидация - Color Loss: {avg_val_loss_color:.4f}, Class Loss: {avg_val_loss_class:.4f}, "
              f"SSIM: {avg_val_ssim:.4f}")

        # Сохраняем чекпоинт
        if (epoch + 1) % config.get('save_frequency', 5) == 0:
            checkpoint_path = os.path.join(os.path.dirname(save_path), f'epoch{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Чекпоинт сохранён: {checkpoint_path}")

        # Ранняя остановка
        if no_improve_count >= patience:
            print(f"Раннее завершение обучения: нет улучшений {patience} эпох подряд")
            break

    # Загружаем лучшую модель для сохранения в качестве финальной
    best_model_path = os.path.join(os.path.dirname(save_path), 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Загружена лучшая модель с валидационной ошибкой: {best_val_loss:.4f}")

    # Сохраняем финальную модель
    torch.save(model.state_dict(), save_path)
    print(f"Финальная модель сохранена: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение TintoraAI")
    parser.add_argument("--config", type=str, default="config.yaml", help="Путь к файлу конфигурации")
    args = parser.parse_args()

    train(args.config)
