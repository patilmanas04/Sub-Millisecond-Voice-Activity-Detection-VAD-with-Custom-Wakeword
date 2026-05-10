import asyncio
import edge_tts
import os
import random

WAKEWORD = "Hello Manas"
SAVE_DIR = "./data/positives/tts"
VARIATIONS_PER_VOICE = 3

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

async def generate_diversity():
    voices = await edge_tts.VoicesManager.create()
    english_voices = voices.find(Language="en")

    print(f"Found {len(english_voices)} English voices. Starting generation...")

    for i, voice in enumerate(english_voices):
        voice_short_name = voice["ShortName"]

        for variation in range(VARIATIONS_PER_VOICE):
            filename = f"{SAVE_DIR}/tts_sample_{i:03d}_{variation}_{voice_short_name}.wav"

            rate_val = random.randint(-20, 20)
            pitch_val = random.randint(-10, 10)
            
            rate = f"{rate_val:+d}%" 
            pitch = f"{pitch_val:+d}Hz"

            try:
                communicate = edge_tts.Communicate(WAKEWORD, voice_short_name, rate=rate, pitch=pitch)
                await communicate.save(filename)
            except Exception as e:
                print(f"Skipped {voice_short_name} (Rate: {rate}, Pitch: {pitch}) - Error: {e}")
            
            # 3. CRITICAL: A 0.5-second pause to prevent Microsoft from blocking your IP
            await asyncio.sleep(0.5) 
        
        if i > 0 and i % 5 == 0:
            print(f"Finished {i} voices (Attempted {(i+1) * VARIATIONS_PER_VOICE} samples so far)...")

    print(f"Generation complete!\nTotal of {len(english_voices) * VARIATIONS_PER_VOICE} samples having different variations of '{WAKEWORD}' are generated and saved in the directory: {SAVE_DIR}")

if __name__ == "__main__":
    asyncio.run(generate_diversity())