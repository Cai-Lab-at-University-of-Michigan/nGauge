# Install for Blender on Mac

Using Blender with nGauge requires the installation of nGauge in one of the Python
system paths which are accessible from Blender. In this example, I will show how to
setup nGauge to work with MacOS. All commands listed here were verified
to work on a Apple M1 Pro MacBook running MacOS 12.0.

## Install Python Libraries
Download and install [Blender](https://www.blender.org/download/) by copying the Blender app from the disk image into your
Application folder following their install directions

Run the following commands in the system terminal:
1. `/Applications/Blender.app/Contents/Resources/2.93/python/bin/python3.9 -m ensurepip`
2. `/Applications/Blender.app/Contents/Resources/2.93/python/bin/python3.9 -m pip install ngauge`

This will install `pip` inside of the `Blender` copy of `Python`, followed by installing ngauge using this new copy of `pip`.

## Verifying Installation
To verify that nGauge is installed correctly, you only need to import
the library inside of your running Blender installation:
1. Start Blender by selecting it from your Applications folder.
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