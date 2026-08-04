# Voicebox con GPU NVIDIA: despliegue y diagnóstico

Guía de despliegue de [`jamiepine/voicebox`](https://github.com/jamiepine/voicebox)
sobre GPU NVIDIA (probado en H200).

El repositorio original solo contempla dos variantes: **CPU** (por defecto) y
**ROCm** para GPUs AMD, esta última mediante un overlay de Compose. **No existe
soporte para NVIDIA/CUDA.** Este documento cubre la variante faltante.

Está dividido en dos partes:

- **Parte I — Solución permanente.** Cambios en el build para que el despliegue
  quede reproducible con un solo comando. Es lo que se debe aplicar.
- **Parte II — Bitácora de diagnóstico.** Registro de los errores encontrados
  durante la puesta en marcha inicial, con su causa raíz. Sirve como referencia
  de troubleshooting y como justificación de cada cambio de la Parte I.

---

## Índice

**Parte I — Solución permanente**

1. [Parche al Dockerfile](#1-parche-al-dockerfile)
2. [Nuevo archivo docker-compose.cuda.yml](#2-nuevo-archivo-docker-composecudayml)
3. [Selección CPU / GPU según el host](#3-selección-cpu--gpu-según-el-host)
4. [Aplicación y verificación](#4-aplicación-y-verificación)
5. [Acceso a la interfaz web vía túnel SSH](#5-acceso-a-la-interfaz-web-vía-túnel-ssh)

**Parte II — Bitácora de diagnóstico**

6. [Error: modelo cargado en CPU](#6-error-modelo-cargado-en-cpu)
7. [Error: sox not found](#7-error-sox-not-found)
8. [Error: número de canales de audio](#8-error-número-de-canales-de-audio)
9. [Error: failed to find C compiler](#9-error-failed-to-find-c-compiler)
10. [Error: permisos en la carpeta de salida](#10-error-permisos-en-la-carpeta-de-salida)
11. [Notas sobre el repositorio original](#11-notas-sobre-el-repositorio-original)

---

# Parte I — Solución permanente

## 1. Parche al Dockerfile

El stage 3 (runtime) del Dockerfile original instala únicamente `ffmpeg`,
`curl` y `gosu`. Faltan `sox` (que `torchaudio` invoca como binario externo) y
un compilador de C (que Triton necesita para compilar kernels en tiempo de
ejecución en GPUs Hopper). `build-essential` sí existe, pero solo en el stage
builder, del que únicamente se copia `/install`.

Sustituir este bloque:

```dockerfile
# Install only runtime system dependencies (gosu drops root in the entrypoint)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    gosu \
    && rm -rf /var/lib/apt/lists/*
```

por:

```dockerfile
# Install only runtime system dependencies (gosu drops root in the entrypoint)
#   sox / libsox-fmt-all : requerido por torchaudio para resamplear y codificar
#                          los audios de referencia de clonación de voz
#   gcc / libc6-dev      : Triton compila kernels de PyTorch en tiempo de
#                          ejecución (JIT) en GPUs NVIDIA modernas (Hopper/H200)
#                          y necesita un compilador de C en el runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    gosu \
    sox \
    libsox-fmt-all \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Triton necesita CC explícito y un caché escribible y persistente
# (mismo criterio que MIOpen en el overlay ROCm)
ENV CC=/usr/bin/gcc \
    TRITON_CACHE_DIR=/app/data/cache/triton
```

Y ampliar el `mkdir` de directorios de datos para incluir el caché de Triton:

```dockerfile
RUN mkdir -p /app/data/generations /app/data/profiles /app/data/cache/triton \
    && chown -R voicebox:voicebox /app/data
```

> Si Triton falla en el enlazado, reemplazar `gcc libc6-dev` por
> `build-essential`. Añade unos ~200 MB a la imagen.

---

## 2. Nuevo archivo `docker-compose.cuda.yml`

Sigue el mismo patrón de overlay que `docker-compose.rocm.yml` del repositorio.

```yaml
---
# CUDA (NVIDIA GPU) overlay para Voicebox
#
#   docker compose -f docker-compose.yml -f docker-compose.cuda.yml up --build -d
#
# Requiere en el host:
#   - Driver NVIDIA
#   - NVIDIA Container Toolkit
#     (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit)
#
# Nota: el stage builder con PYTORCH_VARIANT=cpu instala torch desde PyPI,
# que en linux/amd64 ya trae las wheels con CUDA. No hace falta un
# index-url distinto.

services:
  voicebox:
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      # Reduce la fragmentación del allocator en cargas TTS largas
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    # Memoria compartida para los dataloaders de PyTorch.
    # Los 64 MB por defecto de /dev/shm provocan cuelgues intermitentes.
    shm_size: '2gb'

    deploy:
      resources:
        # Anula los límites de CPU del compose base (cpus 4 / memory 8G).
        # La carga de los safetensors pasa por RAM del host antes de ir a VRAM.
        limits: {}
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

> Si `limits: {}` no anula correctamente los valores del archivo base en tu
> versión de Compose, subir los límites explícitamente en el overlay
> (por ejemplo `cpus: '16'`, `memory: 64G`) en lugar de vaciarlos.

---

## 3. Selección CPU / GPU según el host

La imagen es la misma en ambos casos, no hay que construir dos.
`backend/app.py` detecta el dispositivo en tiempo de ejecución
(`torch.cuda.is_available()` → CUDA o ROCm, luego MPS, luego XPU, y si no,
CPU), así que basta con exponer o no la GPU al contenedor.

El bloque `reservations.devices` con `driver: nvidia` **no** puede vivir en el
compose base, porque en un host sin NVIDIA Container Toolkit el `up` aborta.
Por eso se mantiene en el overlay.

Para no tener que recordar los `-f` en cada comando, definir un `.env` en la
raíz del proyecto, distinto por host:

**Host con GPU NVIDIA** — `.env`:

```dotenv
COMPOSE_FILE=docker-compose.yml:docker-compose.cuda.yml
```

**Host con GPU AMD** — `.env`:

```dotenv
COMPOSE_FILE=docker-compose.yml:docker-compose.rocm.yml
```

**Host solo CPU (laptop, CI, entorno de pruebas)** — `.env` vacío o inexistente.

Con eso, en cualquiera de los tres el comando es idéntico:

```bash
docker compose up -d
```

> `COMPOSE_FILE` usa `:` como separador en Linux/macOS y `;` en Windows.
> El `.env` no debe versionarse; incluir un `.env.example` en su lugar.

### Forzar CPU en un host que sí tiene GPU

Útil para comparar tiempos o descartar que un fallo venga del stack CUDA:

```bash
docker compose run --rm -e CUDA_VISIBLE_DEVICES="" voicebox
```

### Coste de las dependencias añadidas en un host sin GPU

`sox` y `libsox-fmt-all` son necesarios también en CPU: los usa `torchaudio`
para procesar los audios de referencia. `gcc` y `libc6-dev` solo los consume
Triton en GPU, pero suman apenas unas decenas de MB y mantienen una única
imagen para todos los entornos. Si el tamaño importa, se pueden condicionar
con un `ARG INSTALL_JIT_TOOLCHAIN=1` en el stage runtime.

---

## 4. Aplicación y verificación

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up --build -d
```

O simplemente `docker compose up -d --build` si se configuró el `.env` de la
sección anterior.

Verificación:

```bash
# La GPU es visible dentro del contenedor
docker exec voicebox python -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Las dependencias de audio y compilación están presentes
docker exec voicebox sh -c "which sox gcc ffmpeg"
```

### Permisos del bind mount `./output`

El Dockerfile hace `chown -R voicebox:voicebox /app/data`, pero el bind mount
`./output` se monta encima de `/app/data/generations` conservando la propiedad
del host (normalmente `root`). Ejecutar una vez tras el primer arranque:

```bash
docker exec -u 0 voicebox chown -R voicebox:voicebox /app/data
```

El `chown` atraviesa el bind mount, así que la corrección persiste en el host.
Alternativa: eliminar el bind mount `./output` del compose y dejar que el
volumen nombrado `voicebox-data` cubra todo `/app/data`.

---

## 5. Acceso a la interfaz web vía túnel SSH

El compose del repositorio publica el puerto solo en loopback
(`127.0.0.1:17600:17493`), por lo que el servicio no es alcanzable ingresando
la IP pública del servidor en un navegador. Es el comportamiento deseado: se
accede mediante un túnel SSH.

### Con MobaXterm

1. Abrir MobaXterm y presionar **Tunneling** en la barra superior.
2. Hacer clic en **New SSH tunnel**.
3. Seleccionar **Local port forwarding** (primer botón de radio a la izquierda).
4. Completar los campos:

| Campo | Valor requerido |
| :--- | :--- |
| **Forwarded port** *(local)* | `17600` |
| **SSH server** | IP pública del servidor Linux |
| **SSH login** | usuario del servidor |
| **SSH port** | `22` |
| **Remote server** | `127.0.0.1` |
| **Remote port** | `17600` |

5. Hacer clic en **Save** y luego en **Start** (icono de play).
6. Abrir el navegador local e ingresar a `http://localhost:17600`.

### Equivalente por línea de comandos

```bash
ssh -N -L 17600:127.0.0.1:17600 usuario@IP_DEL_SERVIDOR
```

> **Nota de seguridad:** evitar `root` como usuario del túnel. Usar una cuenta
> sin privilegios con autenticación por clave, deshabilitar `PermitRootLogin`
> y restringir el origen del puerto 22 por firewall o grupo de seguridad.

---

# Parte II — Bitácora de diagnóstico

Registro de los errores encontrados durante la puesta en marcha inicial. Los
puntos 7, 9 y 10 ya están resueltos por el parche de la Parte I y **no deben
ejecutarse manualmente** en un despliegue nuevo; se conservan por trazabilidad.

---

## 6. Error: modelo cargado en CPU

**Síntoma**

```
Loading TTS model 1.7B on cpu...
```

seguido del colapso del proceso backend.

**Causa raíz**

El `docker-compose.yml` del repositorio es la variante CPU por defecto y aplica
`deploy.resources.limits` con `cpus: '4'` y `memory: 8G`. Sin devices NVIDIA
expuestos, `torch.cuda.is_available()` devuelve `False` y el modelo
`Qwen3-TTS 1.7B` se carga en RAM, excediendo el límite de 8 GB.

**Resolución**

Overlay `docker-compose.cuda.yml` de la sección 2.

**Estado:** resuelto en Parte I.

---

## 7. Error: `sox not found`

**Síntoma**

```
/bin/sh: 1: sox: not found
SoX could not be found!
```

al procesar un archivo de voz de muestra para clonación.

**Causa raíz**

El stage runtime del Dockerfile instala `ffmpeg`, `curl` y `gosu`, pero no
`sox`. No es un problema del entorno: la dependencia falta en la imagen
original.

**Mitigación temporal aplicada durante la sesión**

```bash
docker exec -u 0 voicebox bash -c "apt-get update && apt-get install -y sox libsox-fmt-all"
```

> Esta instalación es efímera: vive en la capa de escritura del contenedor y se
> pierde con cualquier `docker compose down` o rebuild.

**Estado:** resuelto en Parte I, sección 1.

---

## 8. Error: número de canales de audio

**Síntoma**

```
ValueError: Number of audio channels must be 1 or 2, but got {channels}
```

al procesar audios de muestra grabados con esquemas multicanal.

**Causa raíz**

Limitación de `torchaudio`, que acepta únicamente mono o estéreo. No es un
defecto del despliegue sino del archivo de entrada.

**Resolución**

Convertir los audios de referencia a mono a 24 kHz, escribiendo primero a un
archivo temporal:

```bash
docker exec -u 0 voicebox bash -c \
  'find /app/data/profiles -name "*.wav" -exec sh -c \
   "ffmpeg -y -i \"\$1\" -ac 1 -ar 24000 \"\$1.tmp.wav\" && mv \"\$1.tmp.wav\" \"\$1\"" _ {} \;'
```

> **Importante:** no usar `ffmpeg -y -i {} ... {}` con el mismo archivo como
> entrada y salida. `ffmpeg` lee y escribe simultáneamente y trunca el original.

**Estado:** no requiere cambios en el build. Basta con cargar los perfiles en
formato mono o estéreo estándar.

---

## 9. Error: `failed to find C compiler`

**Síntoma**

```
RuntimeError: Failed to find C compiler. Please specify via CC environment variable.
```

**Causa raíz**

PyTorch usa Triton para compilar kernels en tiempo de ejecución (JIT) en GPUs
NVIDIA modernas. El stage runtime del Dockerfile no incluye compilador de C:
`build-essential` está presente únicamente en el stage builder, del que solo se
copia el directorio `/install`.

**Mitigación temporal aplicada durante la sesión**

```bash
docker exec -u 0 voicebox bash -c "apt-get update && apt-get install -y gcc g++ build-essential"
```

**Estado:** resuelto en Parte I, sección 1, que además fija `CC` y redirige
`TRITON_CACHE_DIR` a un directorio persistente para evitar recompilaciones en
cada recreación del contenedor.

---

## 10. Error: permisos en la carpeta de salida

**Síntoma**

```
OSError: Failed to save audio ...
soundfile.LibsndfileError: Error opening ... .wav.tmp: System error
```

tras completar la inferencia en GPU, al escribir el `.wav` en
`/app/data/generations/`.

**Causa raíz**

El Dockerfile crea y asigna correctamente `/app/data` al usuario `voicebox`,
pero el bind mount `./output` se monta *encima* de `/app/data/generations`
conservando la propiedad del directorio del host, creado por `root`. El montaje
tiene precedencia sobre lo definido en la imagen.

**Resolución**

```bash
docker exec -u 0 voicebox bash -c "mkdir -p /app/data/generations && chown -R voicebox:voicebox /app/data"
```

**Estado:** documentado en Parte I, sección 4. Es inherente al uso de bind
mounts y no se puede resolver enteramente desde el Dockerfile. Evitar
`chmod 777` sobre el directorio de salida.

---

## 11. Notas sobre el repositorio original

Observaciones recogidas al revisar el código fuente, útiles si se abre un PR
upstream:

- **No hay soporte NVIDIA.** Solo existen la variante CPU y el overlay ROCm.
- **El nombre `PYTORCH_VARIANT=cpu` es engañoso.** Esa ruta ejecuta
  `pip install -r requirements.txt` con `torch>=2.2.0` desde PyPI, que en
  `linux/amd64` resuelve a las wheels **con CUDA**. Por eso la GPU funciona sin
  modificar el build: solo faltaba exponer el device al contenedor.
- **El `ENTRYPOINT` es específico de ROCm.** `scripts/rocm-entrypoint.sh` solo
  recorre `/dev/kfd` y `/dev/dri/render*` para unir grupos. Con NVIDIA el bucle
  no hace nada y cae directo a `gosu voicebox`, lo cual es correcto porque el
  NVIDIA Container Toolkit gestiona los devices, pero evidencia que CUDA nunca
  se probó.
- **El overlay ROCm redirige el caché de MIOpen** a `/app/data/cache/` para que
  sea escribible y persistente. El parche de la Parte I replica ese patrón con
  `TRITON_CACHE_DIR`.
- **Volúmenes solapados.** `./output` se monta sobre `/app/data/generations`,
  que a su vez está dentro del volumen nombrado `voicebox-data:/app/data`.
  Funciona, pero el contenido de `generations` en el volumen nombrado queda
  oculto por el bind mount.
- **Falta `shm_size`.** Los 64 MB por defecto de `/dev/shm` son insuficientes
  para los dataloaders multiproceso de PyTorch.

---

## Referencias

- Repositorio: <https://github.com/jamiepine/voicebox>
- NVIDIA Container Toolkit:
  <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit>