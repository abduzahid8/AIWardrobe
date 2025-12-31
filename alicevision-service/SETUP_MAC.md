# Quick Setup for Mac

## Install Dependencies

```bash
cd /Users/zohidvohidjonov/Desktop/AIWardrobe/alicevision-service

# Use pip3 on Mac (not pip)
pip3 install -r requirements.txt
```

## Download Models (Optional - only if you want to test Grounded SAM2)

```bash
# Run the setup script (uses curl instead of wget)
./setup_models.sh
```

## Start Service

```bash
# Use python3 on Mac (not python)
python3 main.py
```

Service will run on http://localhost:5050

## Important Notes

⚠️ **Mac Users**: Use `python3` and `pip3` instead of `python` and `pip`

⚠️ **Model Download**: The models are ~3GB and optional. The service will work without them, but Grounded SAM2 features won't be available until models are downloaded.

⚠️ **GPU Required**: For best performance, Grounded SAM2 needs a GPU. On Mac without NVIDIA GPU, it will use CPU (much slower).

## Quick Test

```bash
# Check if service is running
curl http://localhost:5050/health
```
