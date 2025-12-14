#!/usr/bin/env python3
"""
Test script to monitor NHITS training progress in real-time.

Usage:
    python test_training_progress.py

This script will:
1. Start a training session for a few symbols
2. Poll the progress endpoint every 2 seconds
3. Display real-time progress updates
"""

import asyncio
import httpx
import time
from datetime import datetime


API_BASE = "http://localhost:3011/api/v1"


async def start_training(symbols=None):
    """Start batch training."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{API_BASE}/forecast/train-all",
            params={
                "background": True,
                "force": False
            },
            json=symbols
        )
        response.raise_for_status()
        return response.json()


async def get_progress():
    """Get current training progress."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{API_BASE}/forecast/training/progress")
        response.raise_for_status()
        return response.json()


async def monitor_progress(poll_interval=2):
    """Monitor training progress until completion."""
    print("\n" + "="*80)
    print("NHITS Training Progress Monitor")
    print("="*80 + "\n")

    while True:
        try:
            progress = await get_progress()

            if not progress.get("training_in_progress"):
                print("\n✓ No training in progress")
                if progress.get("last_training_run"):
                    print(f"  Last run: {progress['last_training_run']}")
                break

            # Clear screen (optional - comment out if not desired)
            # print("\033[2J\033[H")

            current_time = datetime.now().strftime("%H:%M:%S")
            current = progress.get("current_symbol", "N/A")
            total = progress.get("total_symbols", 0)
            completed = progress.get("completed_symbols", 0)
            remaining = progress.get("remaining_symbols", 0)
            pct = progress.get("progress_percent", 0)

            results = progress.get("results", {})
            successful = results.get("successful", 0)
            failed = results.get("failed", 0)
            skipped = results.get("skipped", 0)

            timing = progress.get("timing", {})
            elapsed = timing.get("elapsed_formatted", "0s")
            eta = timing.get("eta_formatted", "N/A")

            cancelling = progress.get("cancelling", False)

            # Display progress bar
            bar_width = 50
            filled = int(bar_width * pct / 100)
            bar = "█" * filled + "░" * (bar_width - filled)

            print(f"\n[{current_time}] Training Progress")
            print(f"{'─'*80}")
            print(f"Current Symbol:  {current}")
            print(f"Progress:        [{bar}] {pct}%")
            print(f"Symbols:         {completed}/{total} completed ({remaining} remaining)")
            print(f"")
            print(f"Results:")
            print(f"  ✓ Successful:  {successful}")
            print(f"  ✗ Failed:      {failed}")
            print(f"  ⊘ Skipped:     {skipped}")
            print(f"")
            print(f"Timing:")
            print(f"  Elapsed:       {elapsed}")
            print(f"  ETA:           {eta}")

            if cancelling:
                print(f"\n⚠ CANCELLING - Will stop after current symbol")

            print(f"{'─'*80}\n")

            # Poll again after interval
            await asyncio.sleep(poll_interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")
            await asyncio.sleep(poll_interval)


async def main():
    """Main function."""
    print("NHITS Training Progress Test")
    print("="*80)

    # Check if training is already running
    progress = await get_progress()

    if progress.get("training_in_progress"):
        print("\n✓ Training already in progress!")
        print("  Starting monitoring...\n")
    else:
        print("\n→ No training in progress")
        print("  You can start training with:")
        print(f"    curl -X POST {API_BASE}/forecast/train-all?background=true")
        print("\n  Or run this script to start monitoring, then start training manually.")
        print("\n  Press Ctrl+C to exit")

    # Monitor progress
    await monitor_progress(poll_interval=2)

    print("\n✓ Monitoring complete\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n✓ Stopped by user\n")
