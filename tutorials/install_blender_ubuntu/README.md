# Install for Blender on Ubuntu

Using Blender with nGauge requires the installation of nGauge in one of the Python
system paths which are accessible from Blender. In this example, I will show how to
setup nGauge to work with Ubuntu 20.04 LTS. All commands listed here were verified
to work on a clean install of Ubuntu with updated standard libraries in October of 2021.

## Install Python Libraries
Run the following commands in the system terminal:
1. `sudo apt install blender`
2. `sudo apt install python3-pip`
3. `pip install ngauge`

## Verifying Installation
To verify that nGauge is installed correctly, you only need to import
the library inside of your running Blender installation:
1. Start Blender. This can be done in the command line using the `blender` command with no parameters or by selecting Blendre from you application start menu.
2. Select the "scripting" tab from the top menu of Blender, by default it is the last tab listed.
3. On the left hand side of the window, there should be a Python terminal where commands can be typed.
4. Type `import ngauge` and hit enter.
5. If the command returns nothing, the installation was successful.
6. If you get an error like the one below, something has gone wrong. Try double checking the above commands or following one of the other installation methods given in the project README.

```
>>> import ngauge
Traceback (most recent call last):
  File "/usr/lib/python3.8/code.py", line 90, in runcode
    exec(code, self.locals)
  File "<blender_console>", line 1, in <module>
ModuleNotFoundError: No module named 'ngauge'
```