from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.errors import DistutilsSetupError
from distutils import log as distutils_logger
import subprocess
from hashlib import sha256
import platform
import shutil
import urllib.request
from pathlib import Path
import os
import zipfile

import numpy

def sha256_hash(filename):
    hash = sha256()
    with open(filename, 'rb' ) as fd:
        data_chunk = fd.read(1024)
        while data_chunk:
            hash.update(data_chunk)
            data_chunk = fd.read(1024)

    return hash.hexdigest()

with open("README.md" ,"r", encoding='utf8') as fh:
    long_description = fh.read()

#subprocess.run(['ghcup', 'run', '--stack', '2.7.5', '--', 'stack', 'build', '--force-dirty'], cwd=Path('src/libfcs_ext/hs_submodule'))
# Locate the library and include directories
#built_dynamic_libraries = list(Path('src/libfcs_ext/hs_submodule/.stack-work').glob('**/install/**/*.dll'))
#built_helper_a = list(Path('src/libfcs_ext/hs_submodule/.stack-work').glob('**/*.dll.a'))
#for helper_a in built_helper_a:
#    shutil.copy(helper_a, helper_a.parent / (helper_a.name + '.lib'))
#header_files = list(Path('src/libfcs_ext/hs_submodule/.stack-work').glob('**/install/**/fcs.h'))
#print(built_dynamic_libraries)
#print(header_files)


libfcs_ext = Extension(
    '_libfcs_ext',
    sources=['src/libfcs_ext/libfcs.c'],
    #runtime_library_dirs=['src/libfcs_ext/libfcs/.stack-work/install/47bedf8b/lib'],
#    libraries=[str(x.name) for x in built_helper_a],
#    library_dirs=[str(x.parent) for x in built_helper_a],
#    include_dirs=[str(header_files[0].parent), numpy.get_include()]
)

# From https://downloads.haskell.org/~ghcup/0.1.18.0/SHA256SUMS
ghcup_sha256 = {
    'aarch64-apple-darwin-ghcup-0.1.18.0':'2d3aa19d6f012c1a4ebc5907a05b06cf0d43a1499107020f59847ea2638c8649',
    'aarch64-linux-ghcup-0.1.18.0':'3e3ee4aa06e426373fb7e29f1770987ca1621e414925f261f325f9acb77e0bcb',
    'armv7-linux-ghcup-0.1.18.0':'2e94920c772bc24c9fe41004dedf46840f5f036d28d3ed183679d3f34d2c50e0',
    'i386-linux-ghcup-0.1.18.0':'222914720135261dcc644155bc8a5b15d1d4966c769d50686fe4f41547208759',
    'x86_64-apple-darwin-ghcup-0.1.18.0':'b34ed98bc0cc6d2169974f8d03173f93c3e3f60607013f1af42c9882c1a0d6f0',
    'x86_64-freebsd12-ghcup-0.1.18.0':'cc8378a53f3028331dc853acfb253e2258d720b0e18b618b294ed67182a7fa03',
    'x86_64-freebsd13-ghcup-0.1.18.0':'cc8378a53f3028331dc853acfb253e2258d720b0e18b618b294ed67182a7fa03',
    'x86_64-linux-ghcup-0.1.18.0':'94559eb7c4569919446af1597d07675e803c20b150323edb7f9d8601c8bbda50',
    'x86_64-mingw64-ghcup-0.1.18.0.exe':'e2166a50437c677dfab3362749f676f92ff786aae1bfd7a2d289efa3544ee654'
}

class haskell_dependent_ext(build_ext, object):
    """
    Builder for extensions that require the Haskell libfcs
    module to be built.

    Inspired by https://stackoverflow.com/a/48641638. My modifications
    to this function are available under CC-BY-SA-4.0
    """
    haskell_requiring_extensions = ['_libfcs_ext']
    def build_extension(self, ext):
        print(ext.name)
        if ext.name not in self.haskell_requiring_extensions:
            # Build as normal for any non-haskell-requiring extension
            super(haskell_dependent_ext, self).build_extension(ext)
            return
        # We need to build a Haskell-dependent binary!
        hs_scratch = Path(__file__).parent/'.hsbuild'
        if not hs_scratch.exists():
            hs_scratch.mkdir()

        # Step one: download ghcup if not already downloaded
        sys_arch = platform.machine()
        if platform.system() == 'Linux':
            sys_os = 'linux'
            sys_suffix = ''
        ghcup_binary_name = f'{sys_arch}-{sys_os}-ghcup-0.1.18.0{sys_suffix}'
        ghcup_binary = hs_scratch/ghcup_binary_name
        if not ghcup_binary.exists():
            distutils_logger.info(f"Local ghcup not present. Downloading to {ghcup_binary}")
            r = urllib.request.urlopen(f'https://downloads.haskell.org/~ghcup/0.1.18.0/{ghcup_binary_name}')
            with ghcup_binary.open('wb') as f:
                f.write(r.read())
                ghcup_binary.chmod(0o755)
        if sha256_hash(ghcup_binary) != ghcup_sha256[ghcup_binary_name]:
            raise DistutilsSetupError("Downloaded ghcup appears corrupted! You may want to remove this executable or the entire .hsbuild folder")
        
        install_env = dict(os.environ)|{
            'GHCUP_INSTALL_BASE_PREFIX':str(hs_scratch),
            'GHCUP_SKIP_UPDATE_CHECK':'',
            'PATH':os.environ['PATH']+f':{(hs_scratch/".ghcup"/"bin").resolve()}'
        }

        # Step two: setup Stack/GHC.
        if (not subprocess.run(
                [ghcup_binary, 'list', '-t' ,'stack', '-c', 'installed', '-r'],
                env=install_env, capture_output=True).stdout.decode().startswith('stack 2.7.5')):
            distutils_logger.info(f"Installing stack 2.7.5 into .hsbuild")
            if subprocess.run([ghcup_binary, 'install', 'stack', '2.7.5'], env=install_env).returncode != 0:
                raise DistutilsSetupError("Unable to install stack using local ghcup!")
        else:
            distutils_logger.info(f"Using existing stack 2.7.5")

        if platform.system() == 'Linux':
            # If on Linux, we need to build and patch a -fPIC build of everything
            # Inspired by https://www.hobson.space/posts/haskell-foreign-library/
            # Start by installing a bootstrap GHC
            distutils_logger.info(f"Creating patched GHC version")
            if (not subprocess.run(
                    [ghcup_binary, 'list', '-t' ,'ghc', '-c', 'installed', '-r'],
                    env=install_env, capture_output=True).stdout.decode().startswith('ghc 8.10.7')):
                distutils_logger.info(f"Installing GHC 8.10.7 into .hsbuild")
                if subprocess.run([ghcup_binary, 'install', 'ghc', '8.10.7'], env=install_env).returncode != 0:
                    raise DistutilsSetupError("Unable to install GHC 8.10.7 using local ghcup!")
            else:
                distutils_logger.info(f"Using existing GHC 8.10.7")
            # Only needed if you want to check what you are patching
            #bootstrap_ghc_zip = hs_scratch/'ghc-8.10.7.zip'
            #bootstrap_ghc = hs_scratch/'bootstrap_ghc'
            #if not bootstrap_ghc_zip.exists():
            #    ghc_r = urllib.request.urlopen(f'https://gitlab.haskell.org/ghc/ghc/-/archive/ghc-8.10.7-release/ghc-ghc-8.10.7-release.zip')
            #    with (hs_scratch/'ghc-8.10.7.zip').open('wb') as f:
            #        f.write(ghc_r.read())
            #if sha256_hash(bootstrap_ghc_zip) != '31a824d7f2be69630886095bb68cea8dd062086104f39c251180680f14c0acb4':
            #    raise DistutilsSetupError("Downloaded ghc zip appears corrupted!")
            #
            #if not bootstrap_ghc.exists():
            #    bootstrap_ghc.mkdir()
            #    with zipfile.ZipFile(bootstrap_ghc_zip, 'r') as ghc_zip:
            #        ghc_zip.extractall(bootstrap_ghc)
            linux_pic_dir = hs_scratch.parent/'src'/'libfcs_ext'/'linux_ghc_build'
            subprocess.run(
                [ghcup_binary, 'list'], env=install_env
            )
            subprocess.run(
                [ghcup_binary, 'compile', 'ghc', '-j4', '-v', '8.10.7', '-b', '8.10.7',
                 '-p', str(linux_pic_dir/'patches'), '-c', str(linux_pic_dir/'build.mk'),
                 '-o', '8.10.7-fpic', '--', '--with-system-libffi'], env=install_env
            )


            

        # Step three: build the project
        # Step four: locate link-time binaries and pass them to the extension
        
        # Make the C part of the library as normal
        super(haskell_dependent_ext, self).build_extension(ext)
        

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
    #data_files=[('', [str(x) for x in built_dynamic_libraries])],
    cmdclass = {'build_ext': haskell_dependent_ext},
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
