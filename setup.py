from setuptools import setup, Extension
import subprocess
import sys
import shutil
from pathlib import Path

import numpy

with open("README.md" ,"r", encoding='utf8') as fh:
    long_description = fh.read()

subprocess.run(['ghcup', 'run', '--stack', '2.7.5', '--', 'stack', 'build', '--force-dirty'], cwd=Path('src/libfcs_ext/hs_submodule'))
# Locate the library and include directories
built_dynamic_libraries = list(Path('src/libfcs_ext/hs_submodule/.stack-work').glob('**/install/**/*.dll'))
built_helper_a = list(Path('src/libfcs_ext/hs_submodule/.stack-work').glob('**/*.dll.a'))
for helper_a in built_helper_a:
    shutil.copy(helper_a, helper_a.parent / (helper_a.name + '.lib'))
header_files = list(Path('src/libfcs_ext/hs_submodule/.stack-work').glob('**/install/**/fcs.h'))
print(built_dynamic_libraries)
print(header_files)


libfcs_ext = Extension(
    '_libfcs_ext',
    sources=['src/libfcs_ext/libfcs.c', 'src/libfcs_ext/logicle.c', 'src/libfcs_ext/hyperlog.c'],
    #runtime_library_dirs=['src/libfcs_ext/libfcs/.stack-work/install/47bedf8b/lib'],
    libraries=[str(x.name) for x in built_helper_a],
    library_dirs=[str(x.parent) for x in built_helper_a],
    include_dirs=[str(header_files[0].parent), numpy.get_include()]
)

setup(
    name="libfcs",
    version="0.0.1",
    url='https://github.com/meson800/libfcs-python',
    author="Christopher Johnstone",
    author_email="meson800@gmail.com",
    description="A node- and Bokeh-based flow cytometry platform.",
    license='GPLv2+',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["libfcs"],
    ext_modules=[libfcs_ext],
    package_dir={'': 'src'},
    data_files=[('', [str(x) for x in built_dynamic_libraries])],
    entry_points={
        "console_scripts": [
            "fluent=fluent:dispatch_console"
            ],
        },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "Operating System :: OS Independent",
        ],
    python_requires='>=3',
    install_requires=[
        "numpy"
    ]
)
