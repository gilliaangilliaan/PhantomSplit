#define AppName "PhantomSplit"
#define AppVersion "1.0.0"

[Setup]
AppId={{6C1B7C52-3B0E-4E0A-9A55-7D4E2F1C9A31}
AppName={#AppName}
AppVersion={#AppVersion}
DefaultDirName={code:GetInstallDir}
UsePreviousAppDir=yes
DisableDirPage=auto
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=Output
OutputBaseFilename=PhantomSplit-{#AppVersion}-Setup
VersionInfoVersion={#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
SetupIconFile=..\phantomsplit.ico
UninstallDisplayIcon={app}\phantomsplit.ico
UninstallDisplayName={#AppName}
ChangesAssociations=yes

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"

[Files]
Source: "..\phantomsplitnmdx.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\bs_roformer.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\phantomsplit.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\phantomsplit.png"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\modules\bs_roformer\*.py"; DestDir: "{app}\modules\bs_roformer"; Flags: ignoreversion
Source: "requirements.txt"; DestDir: "{app}\installer"; Flags: ignoreversion
Source: "setup_env.ps1"; DestDir: "{app}\installer"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: """{app}\phantomsplitnmdx.py"""; WorkingDir: "{app}"; IconFilename: "{app}\phantomsplit.ico"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: """{app}\phantomsplitnmdx.py"""; WorkingDir: "{app}"; IconFilename: "{app}\phantomsplit.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: """{app}\phantomsplitnmdx.py"""; WorkingDir: "{app}"; Description: "Launch {#AppName}"; Flags: postinstall nowait skipifsilent; Check: EnvReady

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\tools"
Type: filesandordirs; Name: "{app}\.uv-cache"
Type: filesandordirs; Name: "{app}\models"
Type: filesandordirs; Name: "{app}\modules"
Type: filesandordirs; Name: "{app}\__pycache__"
Type: files; Name: "{app}\phantomsplit_config.json"
Type: files; Name: "{app}\phantomsplit.ico"
Type: files; Name: "{app}\phantomsplit.png"

[Code]
var
  EnvOK: Boolean;

function GetInstallDir(Param: String): String;
var
  Key: String;
begin
  Key := 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{6C1B7C52-3B0E-4E0A-9A55-7D4E2F1C9A31}_is1';
  if not (RegQueryStringValue(HKCU, Key, 'Inno Setup: App Path', Result) and DirExists(Result)) then
    if not (RegQueryStringValue(HKCU64, Key, 'Inno Setup: App Path', Result) and DirExists(Result)) then
      if not (RegQueryStringValue(HKCU32, Key, 'Inno Setup: App Path', Result) and DirExists(Result)) then
        Result := ExpandConstant('{localappdata}\{#AppName}');
end;

function EnvReady: Boolean;
begin
  Result := EnvOK;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    WizardForm.StatusLabel.Caption := 'Installing Python and the PyTorch build for your graphics card (several GB, please wait)...';
    EnvOK := Exec('powershell.exe',
      '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\installer\setup_env.ps1') + '"',
      '', SW_SHOW, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
    if not EnvOK then
      MsgBox('The Python environment could not be set up, so PhantomSplit will not start yet.' + #13#10 + #13#10 +
        'Check your internet connection and run this installer again.', mbError, MB_OK);
  end;
end;
