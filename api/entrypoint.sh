#!/bin/sh
set -e

echo "Fair Value API Starting..."

if [ ! -f "model/final_model_fairvalue.pkl" ]; then
    echo "Model not found. Waiting..."
    while [ ! -f "model/final_model_fairvalue.pkl" ]; do
        sleep 2
    done
    echo "Model detected!"
fi

exec gunicorn \
    --bind 0.0.0.0:5000 \
    --workers 2 \
    --threads 4 \
    --worker-class gthread \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    app:app