import os
import sys
import time
import datetime
import subprocess
import argparse
from train_padufes import run_padufes_pipeline

def trigger_shutdown_if_before_8am(delay_minutes=10):
    """
    Checks the local system time.
    If the current time is before 8:00 AM in the morning,
    triggers a Windows shutdown with the specified delay (in minutes).
    Otherwise, informs the user and skips shutdown.
    """
    now = datetime.datetime.now()
    current_time_str = now.strftime("%Y-%m-%d %I:%M:%S %p")
    print("\n" + "=" * 70)
    print(f"[SHUTDOWN CHECK] Pipeline completed at: {current_time_str}")
    print(f"[SHUTDOWN CHECK] Current hour: {now.hour} (0-23 format)")
    print("=" * 70)
    
    # Check if time is before 8:00 AM (i.e., hour 0, 1, 2, 3, 4, 5, 6, 7)
    if now.hour < 8:
        delay_seconds = delay_minutes * 60
        msg = f"PAD-UFES-20 all 3 models trained successfully before 8:00 AM. Shutting down in {delay_minutes} minutes. Run 'shutdown /a' in command prompt to cancel."
        print(f"\n[ALERT] Current time is BEFORE 8:00 AM.")
        print(f"[ALERT] Triggering PC shutdown in {delay_minutes} minutes ({delay_seconds} seconds)...")
        print(f"[ALERT] To cancel the shutdown anytime, open cmd and run: shutdown /a")
        
        try:
            cmd = ["shutdown", "/s", "/t", str(delay_seconds), "/c", msg]
            subprocess.run(cmd, check=True)
            print(f"[SUCCESS] Windows shutdown scheduled for {delay_minutes} minutes from now.")
        except Exception as e:
            print(f"[WARNING] Failed to schedule shutdown: {e}")
    else:
        print(f"\n[INFO] Current time is {current_time_str} (>= 8:00 AM).")
        print("[INFO] Automatic shutdown skipped since the time is not before 8:00 AM.")

def run_sequential_pipeline(data_dir="PAD-UFES-20", epochs=50, batch_size=32, lr=1e-4, patience=5, start_epoch=25, enable_shutdown=True, shutdown_delay=10):
    print("=" * 75)
    print("      PAD-UFES-20 TRIPLE-MODEL SEQUENTIAL TRAINING PIPELINE")
    print("   Order: [1] EfficientNetV2-S -> [2] Swin Transformer V2-S -> [3] DINOv2-S")
    print("=" * 75)
    
    start_total_time = time.time()
    models_to_train = [
        ("efficientnet", "EfficientNetV2-S"),
        ("swin", "Swin Transformer V2-S"),
        ("dinov2", "DINOv2-S (ViT Foundation)")
    ]
    
    results = {}
    
    for idx, (m_type, m_name) in enumerate(models_to_train, 1):
        print("\n" + "#" * 75)
        print(f"   STEP {idx}/3: TRAINING MODEL '{m_name.upper()}' ON PAD-UFES-20")
        print("#" * 75 + "\n")
        
        m_start = time.time()
        try:
            success = run_padufes_pipeline(
                data_dir=data_dir,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                patience=patience,
                start_epoch=start_epoch,
                model_type=m_type
            )
            duration = time.time() - m_start
            results[m_name] = {"status": "SUCCESS" if success else "FAILED", "duration_min": duration / 60.0}
            print(f"\n>>> Step {idx}/3 ({m_name}) completed in {duration/60.0:.2f} minutes.")
        except Exception as e:
            duration = time.time() - m_start
            results[m_name] = {"status": f"ERROR: {e}", "duration_min": duration / 60.0}
            print(f"\n[ERROR] Model {m_name} encountered an error: {e}")
            
    total_min = (time.time() - start_total_time) / 60.0
    print("\n" + "=" * 75)
    print("              PAD-UFES-20 SEQUENTIAL TRAINING SUMMARY")
    print("=" * 75)
    for m_name, info in results.items():
        print(f"  * {m_name:30s}: {info['status']} ({info['duration_min']:.2f} min)")
    print(f"  Total Duration: {total_min:.2f} minutes ({total_min/60.0:.2f} hours)")
    print("=" * 75)
    
    if enable_shutdown:
        trigger_shutdown_if_before_8am(delay_minutes=shutdown_delay)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet, Swin, and DINOv2 sequentially on PAD-UFES-20 with conditional auto-shutdown.")
    parser.add_argument('--data_dir', type=str, default="PAD-UFES-20", help="PAD-UFES-20 dataset directory")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs per model")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    parser.add_argument('--start_epoch', type=int, default=25, help="Epoch at which early stopping becomes active")
    parser.add_argument('--no_shutdown', action='store_true', help="Disable automatic shutdown check")
    parser.add_argument('--shutdown_delay', type=int, default=10, help="Shutdown delay in minutes (default: 10)")
    args = parser.parse_args()
    
    run_sequential_pipeline(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        start_epoch=args.start_epoch,
        enable_shutdown=not args.no_shutdown,
        shutdown_delay=args.shutdown_delay
    )
