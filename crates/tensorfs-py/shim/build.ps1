# varena's torch allocator shim on Windows (shim/build.sh is Linux's): MSVC against torch 2.14.1's
# headers and c10.lib, from crates/tensorfs-py, before the wheel build. Writes
# python/tensorfs/_varena_torch.dll (c10.dll resolves to the copy torch loaded) and its build record.
$ErrorActionPreference = 'Stop'
$torch = '2.14.1'
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath
Import-Module "$vs\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
$venv = Join-Path $env:RUNNER_TEMP 'varena-shim'
python -m venv $venv
& "$venv\Scripts\python.exe" -m pip install -q --index-url https://download.pytorch.org/whl/cpu "torch==$torch"
$site, $abi, $minor = (& "$venv\Scripts\python.exe" -c 'import os, torch; print(os.path.dirname(torch.__file__), int(torch._C._GLIBCXX_USE_CXX11_ABI), ".".join(torch.__version__.split(".")[:2]))').Split(' ')
cl /nologo /O2 /EHsc /MD /std:c++20 /DNOMINMAX /I"$site\include" shim\varena_torch.cpp /LD /Fe:python\tensorfs\_varena_torch.dll /link /LIBPATH:"$site\lib" c10.lib
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Remove-Item python\tensorfs\_varena_torch.lib, python\tensorfs\_varena_torch.exp, varena_torch.obj -ErrorAction SilentlyContinue
Set-Content -Path python\tensorfs\_varena_torch.json -Value "{`"torch`": `"$minor`", `"cxx11_abi`": $abi}"
