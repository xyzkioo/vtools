; vtools Windows 安装包脚本（Inno Setup 6）。
; 不要直接编译本文件：入口是 packaging/build_windows_installer.py，
; 它会先补齐图标、把路径与版本作为 /D 宏传入，再用 ISCC 生成安装包。
; 前置条件：packaging/build_desktop.py 已生成 dist/vtools/ 目录包。

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

#ifndef SourceDir
  #define SourceDir "..\dist\vtools"
#endif

#ifndef OutputDir
  #define OutputDir "..\dist"
#endif

#ifndef IconFile
  #define IconFile "vtools.ico"
#endif

#define AppName "vtools 工作台"
#define AppExeName "vtools.exe"
#define AppUrl "https://github.com/xyzkioo/vtools"

[Setup]
; 固定 AppID，后续版本按同一路径升级安装，不会出现两个条目。
AppId={{8F3C2E1A-5B47-4C9E-9E1C-2B7E6D4A0F31}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=xyzkioo
AppPublisherURL={#AppUrl}
AppSupportURL={#AppUrl}/issues
AppUpdatesURL={#AppUrl}/releases
VersionInfoVersion={#AppVersion}
VersionInfoDescription={#AppName} 安装程序
VersionInfoProductName={#AppName}
VersionInfoProductVersion={#AppVersion}
VersionInfoCompany=xyzkioo
DefaultDirName={autopf}\vtools
; 升级也显示安装目录，便于从受限目录改回默认用户目录。
DisableDirPage=no
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
AllowNoIcons=yes
; 仅当前用户安装，程序与同目录数据均使用当前用户权限。
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=vtools-{#AppVersion}-windows-x64-setup
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName} {#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
#ifdef ChineseIsl
Name: "chinesesimplified"; MessagesFile: "{#ChineseIsl}"
#endif

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; 目录包必须整体安装，_internal/ 里的运行库和工具源码不能只复制 exe。
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#IconFile}"; DestDir: "{app}"; DestName: "vtools.ico"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\vtools.ico"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\vtools.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[Code]
type
  TAclSizeInformation = record
    AceCount, AclBytesInUse, AclBytesFree: DWORD;
  end;

function GetNamedSecurityInfo(Name: String; ObjectType, Information: DWORD;
  Owner, Group, Dacl: DWORD_PTR; var Sacl, Descriptor: DWORD_PTR): DWORD;
  external 'GetNamedSecurityInfoW@advapi32.dll stdcall';
function GetAclInformation(Acl: DWORD_PTR; var Info: TAclSizeInformation;
  Size, InfoClass: DWORD): Boolean;
  external 'GetAclInformation@advapi32.dll stdcall';
function GetAce(Acl: DWORD_PTR; Index: DWORD; var Ace: DWORD_PTR): Boolean;
  external 'GetAce@advapi32.dll stdcall';
function GetSidSubAuthority(Sid: DWORD_PTR; Index: DWORD): DWORD_PTR;
  external 'GetSidSubAuthority@advapi32.dll stdcall';
procedure CopyDword(var Value: DWORD; Source, Size: DWORD_PTR);
  external 'RtlMoveMemory@kernel32.dll stdcall';
function LocalFree(Memory: DWORD_PTR): DWORD_PTR;
  external 'LocalFree@kernel32.dll stdcall';

function CheckInstallDirectory(): String;
var
  Path: String;
  Sacl, Descriptor, Ace, RidPointer: DWORD_PTR;
  Code, Header, Rid, Index: DWORD;
  Info: TAclSizeInformation;
begin
  Result := '';
  Path := WizardDirValue;
  Log('检查所选安装目录：' + Path);
  while not DirExists(Path) do begin
    if ExtractFileDir(Path) = Path then begin
      Result := '无法访问所选安装目录。';
      Exit;
    end;
    Path := ExtractFileDir(Path);
  end;
  Sacl := 0;
  Descriptor := 0;
  { LABEL_SECURITY_INFORMATION; no ACL or security policy is modified. }
  Code := GetNamedSecurityInfo(Path, 1, $10, 0, 0, 0, Sacl, Descriptor);
  if Code <> 0 then begin
    Result := Format('无法检查安装目录的 Windows 安全属性（错误 %d）：%s', [Code, Path]);
    Exit;
  end;
  try
    if Sacl = 0 then Exit;
    if not GetAclInformation(Sacl, Info, SizeOf(Info), 2) then begin
      Result := '无法读取安装目录的 Windows 安全标签。';
      Exit;
    end;
    if Info.AceCount = 0 then Exit;
    for Index := 0 to Info.AceCount - 1 do begin
      if GetAce(Sacl, Index, Ace) then begin
        CopyDword(Header, Ace, 4);
        if (Header and $FF) = 17 then begin
          RidPointer := GetSidSubAuthority(Ace + 8, 0);
          CopyDword(Rid, RidPointer, 4);
          Log(Format('安装目录完整性检查：%s，RID=%d', [Path, Rid]));
          if Rid < $2000 then begin
            Result := '所选目录被 Windows 标记为低完整性受限位置，无法运行桌面渲染组件。' + #13#10 +
              '安装尚未开始。请选择普通应用程序目录；不要求安装到 C 盘。' + #13#10 + Path;
            Exit;
          end;
        end;
      end;
    end;
  finally
    LocalFree(Descriptor);
  end;
end;

function SetEnvironmentVariable(Name, Value: String): Boolean;
  external 'SetEnvironmentVariableW@kernel32.dll stdcall';

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  StagingDir, OriginalState, StateDir: String;
  Code: Integer;
begin
  Result := CheckInstallDirectory();
  if Result <> '' then Exit;
  OriginalState := GetEnv('VTOOLS_USER_DATA');
  StateDir := OriginalState;
  if StateDir = '' then StateDir := ExpandConstant('{app}\user-data');
  try
    Log('正在准备并运行真实页面安装验收。');
    // Extract the same resources before changing application files or shortcuts.
    ExtractTemporaryFiles('{app}\*');
    StagingDir := ExpandConstant('{tmp}') + '\{app}';
    if not SetEnvironmentVariable('VTOOLS_USER_DATA', StateDir) then begin
      Result := '无法准备安装验收环境，安装尚未开始。';
      Exit;
    end;
    try
      if not Exec(StagingDir + '\vtools-worker.exe', '--install-check',
        StagingDir, SW_HIDE, ewWaitUntilTerminated, Code) then
        Result := '无法启动桌面组件验收，安装尚未开始。'
      else if Code <> 0 then
        Result := '工作台未通过真实页面和用户数据目录验收，安装尚未开始。' + #13#10 +
          '可用时详情日志位于：' + StateDir + '\logs\desktop.log'
      else
        Log('真实 React 页面、本机 API 和原生桥接验收通过，可以开始安装。');
    finally
      SetEnvironmentVariable('VTOOLS_USER_DATA', OriginalState);
    end;
  except
    Result := '安装验收未完成，安装尚未开始：' + GetExceptionMessage;
  end;
end;
