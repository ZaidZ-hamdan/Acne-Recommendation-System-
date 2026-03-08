# Install Python 3.11 to Train the Model

Your PC has **Python 3.14** only. **TensorFlow does not support 3.14**, so training will fail until you add Python 3.11.

## Option A: Install Python 3.11 (recommended)

1. **Download Python 3.11**
   - https://www.python.org/downloads/release/python-3119/
   - Click **"Windows installer (64-bit)"**

2. **Run the installer**
   - Check **"Add python.exe to PATH"**
   - Click **"Install Now"** (or "Customize" and then install)
   - Finish the install

3. **Verify**
   - Open a **new** Command Prompt or PowerShell.
   - Run: `py -3.11 --version`  
     You should see: `Python 3.11.x`

4. **Train the model**
   ```bash
   cd c:\Users\Zaidz\Downloads\CRS
   py -3.11 -m pip install -r requirements.txt
   py -3.11 train_model.py
   ```

You can keep Python 3.14 for other projects; use **3.11 only for this CRS project** (TensorFlow).

## Option B: Use the batch file after installing 3.11

After Python 3.11 is installed, edit `run_training.bat` and change the line that runs Python to use `py -3.11` instead of the 3.14 path, then double‑click the batch file.

Or run in terminal:
```bash
cd c:\Users\Zaidz\Downloads\CRS
py -3.11 -m pip install -r requirements.txt
py -3.11 train_model.py
```
