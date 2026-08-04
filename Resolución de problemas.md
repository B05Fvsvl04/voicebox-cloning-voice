# Guía de solución de problemas y despliegue: Voicebox con GPU NVIDIA H200

Este documento contiene la recopilación completa de los errores detectados durante el despliegue del contenedor de **Voicebox**, sus causas raíz y las soluciones aplicadas para habilitar la aceleración por GPU (NVIDIA H200), la gestión de red mediante túneles SSH y la corrección de dependencias internas.

---

## Índice

1. [Paso 1: Acceso a la interfaz web vía Túnel SSH (MobaXterm)](#paso-1-acceso-a-la-interfaz-web-vía-túnel-ssh-mobaxterm)
2. [Paso 2: Configuración completa de GPU NVIDIA en `docker-compose.yml`](#paso-2-configuración-completa-de-gpu-nvidia-en-docker-composeyml)
3. [Paso 3: Instalación del procesador de audio `SoX`](#paso-3-instalación-del-procesador-de-audio-sox)
4. [Paso 4: Solución a incompatibilidad de canales de audio](#paso-4-solución-a-incompatibilidad-de-canales-de-audio)
5. [Paso 5: Instalación del compilador de C (`gcc`) para Triton / PyTorch](#paso-5-instalación-del-compilador-de-c-gcc-para-triton--pytorch)
6. [Paso 6: Corrección de permisos en la carpeta de salida de audio](#paso-6-corrección-de-permisos-en-la-carpeta-de-salida-de-audio)
7. [Evaluación diaria de avance y aprendizaje](#evaluación-diaria-de-avance-y-aprendizaje)

---

## Paso 1: Acceso a la interfaz web vía Túnel SSH (MobaXterm)

### Problema
El servicio dentro del contenedor expone su puerto mapeado únicamente hacia la interfaz local (`127.0.0.1`), lo que impide el acceso directo ingresando la IP pública del servidor en un navegador convencional.

### Solución detallada
Configurar un túnel de reenvió de puertos (*Local Port Forwarding*) usando **MobaXterm**:

1. Abre MobaXterm y presiona la opción **Tunneling** en la barra superior de herramientas.
2. Haz clic en **New SSH tunnel**.
3. Selecciona la opción **Local port forwarding** (primer botón de radio a la izquierda).
4. Completa los campos con la siguiente configuración exacta:

| Campo | Valor requerido |
| :--- | :--- |
| **Forwarded port** *(Local)* | `17600` |
| **SSH server** | IP pública de tu servidor Linux |
| **SSH login** | `root` |
| **SSH port** | `22` |
| **Remote server** | `127.0.0.1` |
| **Remote port** | `17600` |

5. Haz clic en **Save** y luego presiona el botón **Start** (icono de play) para iniciar el túnel.
6. Abre tu navegador web local e ingresa a: `http://localhost:17600`.

---

## Paso 2: Configuración completa de GPU NVIDIA en `docker-compose.yml`

### Problema
Por defecto, el archivo `docker-compose.yml` limita el contenedor al uso de la CPU (4 núcleos y 8 GB de RAM). Esto provoca que la aplicación intente cargar modelos pesados (como `Qwen3-TTS 1.7B`) en la memoria del procesador (`Loading TTS model 1.7B on cpu...`), provocando el colapso del proceso backend.

### Solución
Reemplazar la totalidad del archivo `docker-compose.yml` ubicado en la carpeta del proyecto (`/home/voicebox/docker-compose.yml`) por la siguiente estructura optimizada para GPUs NVIDIA:

```yaml
# Voicebox — Configuración optimizada para NVIDIA GPU (H200)

services:
  voicebox:
    build: .
    container_name: voicebox
    restart: unless-stopped

    ports:
      # Mapeo del puerto para el túnel SSH (17600 en host -> 17493 interno)
      - "127.0.0.1:17600:17493"

    volumes:
      # Almacenamiento de audio generado
      - ./output:/app/data/generations

      # Persistencia de perfiles y base de datos
      - voicebox-data:/app/data

      # Caché de modelos HuggingFace
      - huggingface-cache:/home/voicebox/.cache/huggingface

    environment:
      - LOG_LEVEL=info
      - NUMBA_CACHE_DIR=/tmp/numba_cache
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility

    networks:
      - voicebox-net

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

networks:
  voicebox-net:
    driver: bridge

volumes:
  voicebox-data:
  huggingface-cache:
