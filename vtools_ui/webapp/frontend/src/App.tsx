import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity, ArrowRight, BarChart3, Boxes, CheckCircle2, ChevronDown, Clock3,
  FileCode2, FileImage, FileText, Folder, FolderOpen, HardDrive, Home, Layers3,
  LoaderCircle, Play, RefreshCw, Search, Settings2, ShieldCheck, Square,
  TerminalSquare, WandSparkles, X, AlertCircle, Database, Copy, ZoomIn, ZoomOut,
  Maximize2, RotateCcw,
} from 'lucide-react'

type Field = { key: string; label: string; kind: string; required?: boolean; default?: unknown; options?: string[]; hint?: string; when?: Record<string, string> }
type Variant = { title: string; fields: Field[]; defaults: Record<string, unknown> }
type Tool = { title: string; description: string; group: string; config?: string; fields?: Field[]; modules?: [string, string][]; common?: string[]; variants?: Record<string, Variant> }
type Catalog = Record<string, Tool>
type RunState = { id: string; tool_id: string; status: string; started_at: string; exit_code: number | null; result_dir: string | null; config: string | null }
type History = { id?: string; time: string; task: string; status: string; config: string; result_dir?: string; exit_code?: number }
type Result = { path: string; relative: string; size: number; kind: 'image' | 'text' | 'model' }
type ConfigField = { path: string; value: unknown; kind: string }
type ConfigDocument = { path: string; text: string; data: Record<string, unknown>; fields: ConfigField[] }
type Settings = { results_root: string; history_limit: number; python_executable: string }
type UpdateInfo = { status: 'unsupported' | 'unpublished' | 'current' | 'available' | 'missing_asset'; current_version: string; latest_version?: string; message: string }
type ResultFolder = { name: string; folders: ResultFolder[]; files: Result[]; count: number }

declare global {
  interface Window { pywebview?: { api: { pick_path: (kind: string) => Promise<string> } } }
}

const token = new URLSearchParams(location.search).get('token') || ''
async function api<T>(path: string, body?: object, method = body ? 'POST' : 'GET'): Promise<T> {
  const response = await fetch(`/api/${path}`, {
    method,
    headers: { 'X-Vtools-Token': token, ...(body ? { 'Content-Type': 'application/json' } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  })
  const payload = await response.json()
  if (!response.ok) throw new Error(payload.error || `请求失败（${response.status}）`)
  return payload as T
}
const q = (value: string) => encodeURIComponent(value)
const active = (state: RunState | null) => Boolean(state && ['starting', 'running', 'stopping'].includes(state.status))
const statusText: Record<string, string> = { starting: '正在启动', running: '运行中', stopping: '正在停止', succeeded: '已完成', succeeded_with_issues: '已完成（有差异/问题）', failed: '运行失败', stopped: '已停止' }
const completionNotice = (state: RunState) => {
  if (state.status === 'succeeded') return state.result_dir ? `任务已完成，结果已保存到：${state.result_dir}` : '任务已完成'
  if (state.status === 'succeeded_with_issues') return state.result_dir ? `任务已完成，发现差异或问题；结果已保存到：${state.result_dir}` : '任务已完成，发现差异或问题'
  if (state.status === 'stopped') return '任务已停止'
  if (state.status === 'failed') return state.exit_code === 127 ? '任务启动失败，请检查 Python 解释器和依赖' : `任务运行失败（退出码 ${state.exit_code ?? '未知'}）`
  return statusText[state.status] || state.status
}
const prettyBackend: Record<string, string> = { pytorch: 'PyTorch 测速', tensorrt: 'TensorRT 测速', consistency: '输出一致性', all: '一键测速', checkpoint: 'Checkpoint 检查' }
const isGoodHistoryStatus = (status: string) => status === '成功'
const isWarningHistoryStatus = (status: string) => ['检查完成（有问题）', '完成（有差异）'].includes(status)

function Icon({ name, size = 19 }: { name: string; size?: number }) {
  const props = { size, strokeWidth: 1.9 }
  switch (name) {
    case 'home': return <Home {...props} />
    case 'diagnostics': return <Activity {...props} />
    case 'visualization': return <Layers3 {...props} />
    case 'benchmark': return <BarChart3 {...props} />
    case 'compression': return <Boxes {...props} />
    case 'transform': return <WandSparkles {...props} />
    case 'data': return <Database {...props} />
    case 'dataset_quality': return <Search {...props} />
    case 'results': return <FolderOpen {...props} />
    case 'history': return <Clock3 {...props} />
    case 'environment': return <ShieldCheck {...props} />
    default: return <Settings2 {...props} />
  }
}

function PathField({ value, onChange, kind, placeholder, hint }: { value: string; onChange: (value: string) => void; kind: string; placeholder?: string; hint?: string }) {
  const [bridgeReady, setBridgeReady] = useState(Boolean(window.pywebview?.api))
  useEffect(() => {
    const ready = () => setBridgeReady(Boolean(window.pywebview?.api))
    window.addEventListener('pywebviewready', ready)
    return () => window.removeEventListener('pywebviewready', ready)
  }, [])
  async function pick(dialogKind: 'file' | 'dir' | 'save_file') {
    if (!window.pywebview?.api) return
    const selected = await window.pywebview.api.pick_path(dialogKind)
    if (selected) onChange(selected)
  }
  const choices: { kind: 'file' | 'dir' | 'save_file'; label: string }[] = kind === 'file_or_dir'
    ? [{ kind: 'file', label: '选择文件' }, { kind: 'dir', label: '选择目录' }]
    : [{ kind: kind === 'dir' ? 'dir' : kind === 'save_file' ? 'save_file' : 'file', label: kind === 'dir' ? '选择目录' : kind === 'save_file' ? '选择保存位置' : '选择文件' }]
  const pathHint = hint || (kind === 'file'
    ? '请选择文件；格式按字段名称要求（如 .pt、.onnx、.yaml、.json、图片或视频）。'
    : kind === 'dir'
      ? '请选择目录；目录内应包含该字段要求的文件。'
      : kind === 'save_file'
        ? '请选择输出文件位置；扩展名按输出格式填写。'
        : '可选择单个文件或目录；文件格式按字段名称要求。')
  return <div className="path-field-group">
    <div className="path-field">
      <input value={value} onChange={event => onChange(event.target.value)} placeholder={placeholder || '输入路径或点击选择'} spellCheck={false} style={{ paddingRight: choices.length === 2 ? 76 : 42 }} />
      {choices.map((choice, index) => <button key={choice.kind} type="button" className="icon-button" style={{ right: 5 + (choices.length - index - 1) * 34 }} title={bridgeReady ? choice.label : '浏览器预览可直接输入路径'} aria-label={choice.label} onClick={() => pick(choice.kind)} disabled={!bridgeReady}>{choice.kind === 'dir' ? <FolderOpen size={18} /> : <FileText size={18} />}</button>)}
    </div>
    {!hint && <small className="path-hint">{pathHint}</small>}
  </div>
}

function InputField({ field, value, onChange }: { field: Field; value: unknown; onChange: (value: unknown) => void }) {
  const kind = field.kind
  if (kind === 'bool') return <label className="switch-row"><input type="checkbox" checked={Boolean(value)} onChange={event => onChange(event.target.checked)} /><span className="switch" /><span>{field.label}</span></label>
  if (kind === 'select' || kind.startsWith('select:')) {
    const options = field.options || kind.slice(7).split('|')
    return <select value={String(value ?? options[0] ?? '')} onChange={event => onChange(event.target.value)}>{options.map(option => <option key={option} value={option}>{prettyBackend[option] || option}</option>)}</select>
  }
  if (['file', 'dir', 'save_file', 'file_or_dir'].includes(kind)) return <PathField value={String(value ?? '')} onChange={onChange} kind={kind} hint={field.hint} />
  return <input type={kind === 'number' ? 'number' : 'text'} value={String(value ?? '')} onChange={event => onChange(event.target.value)} placeholder={field.hint || (field.required ? '必填' : '可选')} />
}

function ResultFolderTree({ node, depth = 0, selectedFile, onSelect }: { node: ResultFolder; depth?: number; selectedFile: Result | null; onSelect: (file: Result) => void }) {
  const children = <>
    {node.folders.map(folder => <ResultFolderTree key={folder.name} node={folder} depth={depth + 1} selectedFile={selectedFile} onSelect={onSelect} />)}
    {node.files.map(file => <button className={`file-row ${selectedFile?.path === file.path ? 'selected' : ''}`} onClick={() => onSelect(file)} key={file.path} style={{ paddingLeft: `${10 + depth * 16}px` }}>
      {file.kind === 'image' ? <FileImage size={17} /> : file.kind === 'model' ? <Boxes size={17} /> : <FileText size={17} />}
      <span title={file.relative}>{file.relative.split(/[\\/]/).at(-1)}</span>
    </button>)}
  </>
  if (node.name === '根目录') return <>{children}</>
  return <details className="result-tree-folder" open={depth < 2}>
    <summary style={{ paddingLeft: `${7 + (depth - 1) * 16}px` }}><Folder size={16} /><span title={node.name}>{node.name}</span><span className="folder-count">{node.count}</span></summary>
    <div className="result-tree-children">{children}</div>
  </details>
}

function ConfigEditor({ path, tool, onClose, onSaved }: { path: string; tool: Tool; onClose: () => void; onSaved: (path: string) => void }) {
  const [doc, setDoc] = useState<ConfigDocument | null>(null)
  const [tab, setTab] = useState<'common' | 'all' | 'yaml'>('common')
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const [drafts, setDrafts] = useState<Record<string, string | boolean>>({})
  useEffect(() => {
    api<ConfigDocument>(`config?path=${q(path)}`).then(setDoc).catch(err => setError(String(err.message)))
  }, [path])

  async function applyDrafts(current: ConfigDocument): Promise<ConfigDocument> {
    let updated = current
    for (const [path, raw] of Object.entries(drafts)) {
      const field = updated.fields.find(item => item.path === path)
      if (!field) continue
      let value: unknown = raw
      if (field.kind === 'number') {
        if (typeof raw !== 'string' || raw.trim() && !Number.isFinite(Number(raw))) throw new Error(`${path} 需要有效数字`)
        value = raw.trim() ? Number(raw) : null
      } else if (field.kind === 'list') value = String(raw).split(',').map(item => item.trim()).filter(Boolean)
      const patched = await api<Omit<ConfigDocument, 'path'>>('config/patch', { text: updated.text, path, value })
      updated = { ...updated, ...patched }
    }
    setDrafts({})
    setDoc(updated)
    return updated
  }
  async function changeTab(next: 'common' | 'all' | 'yaml') {
    if (!doc) return
    let updated = doc
    try { updated = await applyDrafts(updated) }
    catch (err) { setError(String(err)); return }
    if (tab === 'yaml' && doc) {
      try {
        const parsed = await api<{ data: Record<string, unknown>; fields: ConfigField[] }>('config/parse', { text: updated.text })
        setDoc({ ...updated, data: parsed.data, fields: parsed.fields })
        setError('')
      } catch (err) { setError(String(err)); return }
    }
    setTab(next)
  }
  async function save() {
    if (!doc) return
    setSaving(true)
    try {
      const updated = await applyDrafts(doc)
      await api('config/save', { path: updated.path, text: updated.text })
      onSaved(updated.path)
      onClose()
    } catch (err) { setError(String(err)) }
    finally { setSaving(false) }
  }
  const all = doc?.fields || []
  const common = (tool.common || []).map(key => all.find(item => item.path === key)).filter((item): item is ConfigField => Boolean(item))
  const visible = tab === 'common' ? common : all
  return <div className="modal-backdrop" role="presentation" onMouseDown={onClose}><section className="modal" role="dialog" aria-modal="true" aria-label="编辑配置" onMouseDown={event => event.stopPropagation()}>
    <div className="modal-header"><div><div className="eyebrow">CONFIGURATION</div><h2>图形化配置</h2><p className="path-caption">{path}</p></div><button className="icon-button" onClick={onClose} aria-label="关闭"><X size={20} /></button></div>
    <div className="tabs">{([['common', '常用配置'], ['all', '全部字段'], ['yaml', 'YAML 原文']] as const).map(([key, label]) => <button key={key} className={tab === key ? 'selected' : ''} onClick={() => changeTab(key)}>{label}</button>)}</div>
    <div className="modal-content">{!doc ? <p className="muted">正在读取配置…</p> : tab === 'yaml' ? <textarea className="yaml-editor" value={doc.text} onChange={event => setDoc({ ...doc, text: event.target.value })} spellCheck={false} /> : <div className="config-fields">{visible.map(field => <label className="config-field" key={field.path}><span>{field.path}</span>{field.kind === 'bool' ? <input type="checkbox" checked={Boolean(drafts[field.path] ?? field.value)} onChange={event => setDrafts({ ...drafts, [field.path]: event.target.checked })} /> : <input value={String(drafts[field.path] ?? (Array.isArray(field.value) ? field.value.join(', ') : field.value ?? ''))} onChange={event => setDrafts({ ...drafts, [field.path]: event.target.value })} />}</label>)}</div>}</div>
    {error && <div className="inline-error"><AlertCircle size={17} />{error}</div>}
    <div className="modal-footer"><span className="muted small">保存前会校验 YAML。</span><button className="button secondary" onClick={onClose}>取消</button><button className="button primary" onClick={save} disabled={!doc || saving}>{saving ? '保存中…' : '保存配置'}</button></div>
  </section></div>
}

function App() {
  const [catalog, setCatalog] = useState<Catalog>({})
  const [settings, setSettings] = useState<Settings | null>(null)
  const [historyRows, setHistoryRows] = useState<History[]>([])
  const [page, setPage] = useState('home')
  const [search, setSearch] = useState('')
  const [environments, setEnvironments] = useState<{ label: string; path: string }[]>([])
  const [run, setRun] = useState<RunState | null>(null)
  const [log, setLog] = useState('')
  const [logOpen, setLogOpen] = useState(false)
  const [notice, setNotice] = useState('')
  const [configPath, setConfigPath] = useState('')
  const [configData, setConfigData] = useState<Record<string, unknown>>({})
  const [configEditor, setConfigEditor] = useState(false)
  const [values, setValues] = useState<Record<string, unknown>>({})
  const [moduleEdits, setModuleEdits] = useState<Record<string, boolean>>({})
  const [variant, setVariant] = useState('')
  const [files, setFiles] = useState<Result[]>([])
  const [resultRoot, setResultRoot] = useState('')
  const [resultFilter, setResultFilter] = useState('all')
  const [selectedFile, setSelectedFile] = useState<Result | null>(null)
  const [preview, setPreview] = useState('')
  const [imageScale, setImageScale] = useState(1)
  const [imageFit, setImageFit] = useState(true)
  const [resultsBusy, setResultsBusy] = useState(false)
  const [settingsDraft, setSettingsDraft] = useState<Settings | null>(null)
  const [version, setVersion] = useState('')
  const [updateInfo, setUpdateInfo] = useState<UpdateInfo | null>(null)
  const [updateBusy, setUpdateBusy] = useState(false)
  const [updateError, setUpdateError] = useState('')
  const [updateInstalled, setUpdateInstalled] = useState(false)
  const logRef = useRef<HTMLPreElement>(null)

  const tool = catalog[page]
  const isToolPage = Boolean(tool)
  const modelTool = Boolean(tool?.config)
  const isUtility = Boolean(tool?.variants)
  const selectedVariant = isUtility ? tool.variants?.[variant] : undefined

  const refreshHistory = useCallback(() => api<{ history: History[] }>('history').then(data => setHistoryRows(data.history)).catch(err => setNotice(String(err))), [])
  const refreshResults = useCallback(async (root: string) => {
    if (!root) return
    setResultsBusy(true)
    try {
      const data = await api<{ root: string; files: Result[] }>(`results?root=${q(root)}`)
      setResultRoot(data.root); setFiles(data.files); setSelectedFile(null); setPreview('')
    } catch (err) { setNotice(String(err)) }
    finally { setResultsBusy(false) }
  }, [])

  useEffect(() => {
    api<{ catalog: Catalog; settings: Settings; history: History[]; state: RunState | null; version: string }>('bootstrap').then(data => {
      setCatalog(data.catalog); setSettings(data.settings); setSettingsDraft(data.settings)
      setHistoryRows(data.history); setRun(data.state); setResultRoot(data.settings.results_root); setVersion(data.version)
    }).catch(err => setNotice(String(err)))
    api<{ environments: { label: string; path: string }[] }>('environments').then(data => setEnvironments(data.environments)).catch(() => {})
    api<UpdateInfo>('update/check').then(setUpdateInfo).catch(err => setUpdateError(String(err)))
  }, [])

  useEffect(() => {
    if (!tool) return
    const firstVariant = Object.keys(tool.variants || {})[0] || ''
    setValues({ ...Object.fromEntries((tool.fields || []).filter(field => field.default !== undefined).map(field => [field.key, field.default])), ...(firstVariant ? tool.variants?.[firstVariant].defaults || {} : {}) })
    setModuleEdits({})
    setConfigData({})
    setVariant(firstVariant)
    setConfigPath(tool.config || '')
  }, [page, catalog])

  useEffect(() => {
    if (!modelTool || !configPath) return
    let cancelled = false
    api<ConfigDocument>(`config?path=${q(configPath)}`).then(data => {
      if (!cancelled) { setConfigData(data.data); setModuleEdits({}); if (page === 'compression') { const model = data.data.model as Record<string, unknown> | undefined; if (model?.task === 'classify') setValues(previous => ({ ...previous, task: 'classify' })) } }
    }).catch(err => { if (!cancelled) setNotice(String(err)) })
    return () => { cancelled = true }
  }, [configPath, modelTool])

  useEffect(() => {
    if (!run?.id || !active(run)) return
    let cancelled = false
    let cursor = 0
    const poll = async () => {
      while (!cancelled) {
        try {
          const data = await api<{ events: { kind: string; payload: unknown }[]; cursor: number; done: boolean }>(`runs/events?id=${q(run.id)}&cursor=${cursor}`)
          if (cancelled) return
          cursor = data.cursor
          for (const event of data.events) {
            if (event.kind === 'log') setLog(previous => previous + String(event.payload))
            else if (event.kind === 'state') setRun(event.payload as RunState)
          }
          if (data.done) {
            const current = await api<{ state: RunState }>('runs/current')
            if (!cancelled) {
              setRun(current.state)
              setNotice(completionNotice(current.state))
              refreshHistory()
            }
            break
          }
        } catch (err) { setNotice(String(err)); break }
      }
    }
    poll()
    return () => { cancelled = true }
  }, [run?.id, refreshHistory])

  useEffect(() => { if (logOpen && logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight }, [log, logOpen])
  useEffect(() => { if (page === 'results' && resultRoot) refreshResults(resultRoot) }, [page])

  async function start() {
    if (!tool || active(run)) return
    if (page === 'data' && variant === 'filename' && values.apply && !window.confirm('将直接修改文件名。确认继续吗？')) return
    try {
      const payload = await api<{ state: RunState }>('runs', {
        tool_id: page, variant: isUtility ? variant : undefined, config_path: modelTool ? configPath : undefined,
        python_executable: settings?.python_executable, values, module_edits: moduleEdits,
      })
      setRun(payload.state); setLog(''); setLogOpen(true); setNotice('任务已启动')
    } catch (err) { setNotice(String(err)) }
  }
  async function stop() {
    try { const data = await api<{ state: RunState }>('runs/stop', {}); setRun(data.state); setNotice(data.state.status === 'stopped' ? '任务已停止' : '正在停止任务') }
    catch (err) { setNotice(String(err)) }
  }
  async function saveSettings() {
    if (!settingsDraft) return
    try {
      const data = await api<{ settings: Settings }>('settings', settingsDraft)
      setSettings(data.settings); setSettingsDraft(data.settings); setResultRoot(data.settings.results_root); setNotice('设置已保存')
    } catch (err) { setNotice(String(err)) }
  }
  async function checkForUpdates() {
    setUpdateBusy(true); setUpdateError('')
    try { setUpdateInfo(await api<UpdateInfo>('update/check')) }
    catch (err) { setUpdateError(String(err)) }
    finally { setUpdateBusy(false) }
  }
  async function installUpdate() {
    setUpdateBusy(true); setUpdateError('')
    try {
      const result = await api<{ message: string }>('update/install', {})
      setUpdateInstalled(true); setNotice(result.message)
    } catch (err) { setUpdateError(String(err)) }
    finally { setUpdateBusy(false) }
  }
  async function selectFile(file: Result) {
    setSelectedFile(file); setPreview(''); setImageScale(1); setImageFit(true)
    if (file.kind !== 'text') return
    try { const data = await api<{ text: string; truncated: boolean }>(`results/preview?path=${q(file.path)}`); setPreview(data.text + (data.truncated ? '\n\n—— 仅预览前 256 KB ——' : '')) }
    catch (err) { setPreview(String(err)) }
  }
  async function openFile(file: Result) {
    try { await api('results/open', { path: file.path, folder: true }) }
    catch (err) { setNotice(String(err)) }
  }
  function zoomImage(delta: number) {
    setImageFit(false)
    setImageScale(previous => Math.min(4, Math.max(0.25, Number((previous + delta).toFixed(2)))))
  }
  function fitImage() {
    setImageFit(true)
    setImageScale(1)
  }
  function resetImage() {
    setImageFit(false)
    setImageScale(1)
  }
  const visibleFiles = useMemo(() => files.filter(file => resultFilter === 'all' || file.kind === resultFilter), [files, resultFilter])
  const resultTree = useMemo(() => {
    const root: ResultFolder = { name: '根目录', folders: [], files: [], count: 0 }
    for (const file of visibleFiles) {
      const parts = file.relative.split(/[\\/]/).filter(Boolean)
      let node = root
      const ancestors = [root]
      const folderParts = parts.slice(0, -1)
      for (const part of folderParts) {
        let child = node.folders.find(folder => folder.name === part)
        if (!child) { child = { name: part, folders: [], files: [], count: 0 }; node.folders.push(child) }
        node = child
        ancestors.push(node)
      }
      node.files.push(file)
      ancestors.forEach(folder => { folder.count += 1 })
    }
    const sort = (node: ResultFolder) => { node.folders.sort((a, b) => a.name.localeCompare(b.name)); node.files.sort((a, b) => a.relative.localeCompare(b.relative)); node.folders.forEach(sort) }
    sort(root)
    return root
  }, [visibleFiles])

  const navGroups = [
    { label: '工作台', items: [['home', '总览']] },
    { label: '模型工具', items: [['diagnostics', '检测诊断'], ['visualization', '模型可视化'], ['benchmark', '性能测速'], ['compression', '模型压缩']] },
    { label: '数据与转换', items: [['dataset_quality', '数据集质量检查'], ['transform', '格式转换'], ['data', '数据工具']] },
    { label: '记录与设置', items: [['results', '结果查看'], ['history', '任务记录'], ['environment', '环境检查'], ['settings', '设置']] },
  ]
  const pageTitle = page === 'home' ? '工作台' : page === 'results' ? '结果查看' : page === 'history' ? '任务记录' : page === 'settings' ? '设置' : tool?.title || '工作台'
  const moduleMap = (configData.modules || {}) as Record<string, boolean>
  const displayedFields = isUtility ? selectedVariant?.fields || [] : tool?.fields || []
  const fieldValues = isUtility ? { ...(selectedVariant?.defaults || {}), ...values } : values

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><Layers3 size={23} /></div><div><strong>vtools</strong><span>视觉模型工作台</span></div></div>
      <nav>{navGroups.map(group => {
        const items = group.items.filter(([, title]) => !search || title.includes(search) || group.label.includes(search))
        return items.length ? <div className="nav-group" key={group.label}><div className="nav-label">{group.label}</div>{items.map(([key, title]) => <button key={key} className={`nav-item ${page === key ? 'current' : ''}`} onClick={() => setPage(key)}><Icon name={key} /><span>{title}</span>{page === key && <span className="nav-dot" />}</button>)}</div> : null
      })}</nav>
      <div className="sidebar-bottom"><div className="sidebar-tip"><ShieldCheck size={17} /><span>任务在本机运行</span></div><span className="version">vtools · {version || 'Desktop'}</span></div>
    </aside>

    <div className="main-shell">
      <header className="topbar"><div className="breadcrumb">vtools <span>/</span> <strong>{pageTitle}</strong></div><div className="top-actions"><div className="top-search"><Search size={16} /><input value={search} onChange={event => setSearch(event.target.value)} placeholder="搜索工具" aria-label="搜索工具" /></div><div className="env-picker"><HardDrive size={16} /><select value={settings?.python_executable || ''} onChange={event => { const selected = event.target.value; if (settings) { setSettings({ ...settings, python_executable: selected }); setSettingsDraft(previous => previous ? { ...previous, python_executable: selected } : previous); api('settings', { python_executable: selected }).catch(err => setNotice(String(err))) } }}><option value={settings?.python_executable || ''}>{settings?.python_executable ? settings.python_executable.split(/[\\/]/).slice(-3).join(' / ') : '当前 Python'}</option>{environments.filter(env => env.path !== settings?.python_executable).map(env => <option value={env.path} key={env.path}>{env.label} · {env.path}</option>)}</select><ChevronDown size={14} /></div><span className="top-avatar">VT</span></div></header>

      <main className="content">
        {page === 'home' && <>
          <div className="welcome-banner"><div><div className="eyebrow">YOUR WORKSPACE</div><h1>今天想分析什么？</h1><p>从模型诊断到结果对比，让每一步都清晰可见。</p><button className="button light" onClick={() => setPage('diagnostics')}>开始检测诊断 <ArrowRight size={17} /></button></div><div className="banner-art"><div className="orb orb-one" /><div className="orb orb-two" /><Layers3 size={105} strokeWidth={1} /></div></div>
          <div className="section-heading"><div><div className="eyebrow">QUICK START</div><h2>常用工具</h2></div></div>
          <div className="quick-grid">{['diagnostics', 'dataset_quality', 'visualization', 'benchmark', 'compression'].map((key, index) => <button className="quick-card" key={key} onClick={() => setPage(key)}><span className={`quick-icon tone-${index % 4}`}><Icon name={key} size={24} /></span><strong>{catalog[key]?.title || key}</strong><span>{catalog[key]?.description}</span><ArrowRight className="quick-arrow" size={18} /></button>)}</div>
          <div className="section-heading recent-heading"><div><div className="eyebrow">RECENT ACTIVITY</div><h2>最近任务</h2></div><button className="link-button" onClick={() => setPage('history')}>查看全部 <ArrowRight size={16} /></button></div>
          <div className="card recent-list">{historyRows.length ? historyRows.slice(0, 5).map((row, index) => <div className="recent-row" key={row.id || index}><span className={`status-icon ${isGoodHistoryStatus(row.status) ? 'good' : isWarningHistoryStatus(row.status) ? 'warning' : row.status === '失败' ? 'bad' : ''}`}>{isGoodHistoryStatus(row.status) ? <CheckCircle2 size={18} /> : isWarningHistoryStatus(row.status) ? <AlertCircle size={18} /> : <Clock3 size={18} />}</span><div><strong>{catalog[row.task]?.title || row.task}</strong><small>{row.config}</small></div><span className="recent-time">{row.time}</span><span className={`pill ${isGoodHistoryStatus(row.status) ? 'good' : isWarningHistoryStatus(row.status) ? 'warning' : row.status === '失败' ? 'bad' : ''}`}>{row.status}</span></div>) : <div className="empty-state">还没有任务记录。运行第一个工具后会显示在这里。</div>}</div>
        </>}

        {isToolPage && <>
          <div className="page-heading"><div className="eyebrow">{tool.group === 'model' ? 'MODEL WORKFLOW' : tool.group === 'utility' ? 'DATA WORKFLOW' : 'SYSTEM CHECK'}</div><h1>{tool.title}</h1><p>{tool.description}</p></div>
          {isUtility && <div className="card form-card"><div className="card-heading"><span className="step-number">01</span><div><h2>选择操作</h2><p>选择你要运行的具体工具。</p></div></div><div className="variant-grid">{Object.entries(tool.variants || {}).map(([key, item]) => <button className={`variant-card ${variant === key ? 'chosen' : ''}`} key={key} onClick={() => { setVariant(key); setValues({ ...(item.defaults || {}) }) }}><FileCode2 size={20} /><strong>{item.title}</strong></button>)}</div></div>}
          {modelTool && <div className="card form-card"><div className="card-heading"><span className="step-number">01</span><div><h2>配置文件</h2><p>YAML 是完整参数来源，本页只覆盖本次运行所需的输入。</p></div></div><div className="form-grid"><label className="form-field wide"><span>配置文件 <em>必填</em></span><PathField value={configPath} onChange={setConfigPath} kind="file" /></label></div><button className="link-button config-link" onClick={() => setConfigEditor(true)}><Settings2 size={17} /> 图形化编辑配置 <ArrowRight size={16} /></button></div>}
          <div className="card form-card"><div className="card-heading"><span className="step-number">{isUtility ? '02' : modelTool ? '02' : '01'}</span><div><h2>输入与参数</h2><p>路径支持直接输入；桌面窗口可以点击文件夹图标选择。</p></div></div><div className="form-grid">{displayedFields.filter(field => !field.when || Object.entries(field.when).every(([key, expected]) => fieldValues[key] === expected)).map(field => field.kind === 'bool' ? <div className="form-field switch-field" key={field.key}><InputField field={field} value={fieldValues[field.key]} onChange={value => setValues({ ...values, [field.key]: value })} /></div> : <label className={`form-field ${['file', 'dir', 'file_or_dir', 'save_file'].includes(field.kind) ? 'wide' : ''}`} key={field.key}><span>{field.label} {field.required && <em>必填</em>}</span><InputField field={field} value={fieldValues[field.key]} onChange={value => setValues({ ...values, [field.key]: value })} />{field.hint && <small>{field.hint}</small>}</label>)}</div></div>
          {modelTool && <div className="card form-card"><div className="card-heading"><span className="step-number">03</span><div><h2>运行模块</h2><p>{page === 'benchmark' && values.backend === 'checkpoint' ? 'Checkpoint 检查按配置文件运行，不支持本次运行临时切换模块。' : '仅修改这里点击过的模块；配置中未展示的模块保持原样。'}</p></div></div><div className="module-grid">{(tool.modules || []).map(([id, label]) => { const incompatible = (page === 'benchmark' && values.backend === 'checkpoint') || (page === 'compression' && ((values.task === 'detect' && ['compression.quantize.dynamic_int8', 'distillation.classification'].includes(id)) || (values.task === 'classify' && id === 'compression.prune.structured'))); return <label className={`module-option ${incompatible ? 'disabled' : ''}`} key={id} title={incompatible ? '当前任务不支持临时切换此模块' : id}><input type="checkbox" disabled={incompatible} checked={moduleEdits[id] ?? Boolean(moduleMap[id])} onChange={event => setModuleEdits({ ...moduleEdits, [id]: event.target.checked })} /><span className="checkmark" /><span><strong>{label}</strong><small>{id}</small></span></label> })}</div></div>}
          <div className="run-card"><div><span className="run-card-icon"><Play size={20} fill="currentColor" /></span><div><strong>准备就绪</strong><span>确认输入和参数后开始运行。过程和错误会显示在运行详情中。</span></div></div><button className="button primary" onClick={start} disabled={active(run)}>{active(run) ? '任务运行中' : '开始运行'} <ArrowRight size={17} /></button></div>
          {run?.tool_id === page && run.result_dir && !active(run) && <div className="result-callout"><CheckCircle2 size={20} /><span>本次结果：{run.result_dir}</span><button className="link-button" onClick={() => { setResultRoot(run.result_dir!); setPage('results') }}>查看结果 <ArrowRight size={16} /></button></div>}
        </>}

        {page === 'results' && <><div className="page-heading"><div className="eyebrow">OUTPUT LIBRARY</div><h1>结果查看</h1><p>浏览运行目录中的报告、图片和模型产物。</p></div><div className="card results-toolbar"><PathField value={resultRoot} onChange={setResultRoot} kind="dir" /><button className="button secondary" onClick={() => refreshResults(resultRoot)}><RefreshCw size={16} /> {resultsBusy ? '扫描中' : '刷新'}</button></div><div className="result-layout"><div className="card result-list"><div className="result-list-head"><strong>文件列表 <span>{visibleFiles.length}</span></strong><select value={resultFilter} onChange={event => setResultFilter(event.target.value)}><option value="all">全部</option><option value="image">图片</option><option value="text">文档</option><option value="model">模型</option></select></div><div className="result-scroll"><ResultFolderTree node={resultTree} selectedFile={selectedFile} onSelect={selectFile} />{!files.length && <div className="empty-state">当前目录没有可预览的结果文件。</div>}</div></div><div className="card preview-panel">{selectedFile ? <><div className="preview-head"><div><div className="eyebrow">FILE PREVIEW</div><h2>{selectedFile.relative.split(/[\\/]/).at(-1)}</h2><p>{selectedFile.relative} · {(selectedFile.size / 1024).toFixed(1)} KB</p></div><div className="preview-actions"><button className="button secondary" onClick={() => openFile(selectedFile)}><FolderOpen size={16} /> 打开所在目录</button>{selectedFile.kind === 'image' && <div className="image-controls" aria-label="图片缩放"><button className="icon-button" onClick={() => zoomImage(-0.25)} disabled={imageScale <= 0.25} title="缩小" aria-label="缩小"><ZoomOut size={16} /></button><span>{Math.round(imageScale * 100)}%</span><button className="icon-button" onClick={() => zoomImage(0.25)} disabled={imageScale >= 4} title="放大" aria-label="放大"><ZoomIn size={16} /></button><button className={`icon-button ${imageFit ? 'selected' : ''}`} onClick={fitImage} title="适应窗口" aria-label="适应窗口"><Maximize2 size={16} /></button><button className="icon-button" onClick={resetImage} title="重置比例" aria-label="重置比例"><RotateCcw size={16} /></button></div>}</div></div>{selectedFile.kind === 'image' ? <div className={`image-preview ${imageFit ? 'fit' : 'free'}`}><img style={{ transform: `scale(${imageScale})` }} src={`/api/results/preview?path=${q(selectedFile.path)}&token=${q(token)}`} alt={selectedFile.relative} /></div> : selectedFile.kind === 'text' ? <pre className="text-preview">{preview || '正在读取…'}</pre> : <div className="empty-state model-preview"><Boxes size={46} /><strong>模型产物</strong><p>可从所在目录用合适的工具打开此文件。</p></div>}</> : <div className="empty-state preview-empty"><FileImage size={42} /><strong>选择文件查看预览</strong><p>支持图片、JSON、CSV、YAML 和文本。</p></div>}</div></div></>}

        {page === 'history' && <><div className="page-heading"><div className="eyebrow">ACTIVITY LOG</div><h1>任务记录</h1><p>查看每次运行的状态、配置和输出位置。</p></div><div className="card history-card"><div className="history-head"><strong>全部任务 <span>{historyRows.length}</span></strong><button className="link-button" onClick={refreshHistory}><RefreshCw size={16} /> 刷新</button></div><div className="table-wrap"><table><thead><tr><th>时间</th><th>任务</th><th>状态</th><th>配置</th><th>结果</th></tr></thead><tbody>{historyRows.map((row, index) => <tr key={row.id || index}><td>{row.time}</td><td><strong>{catalog[row.task]?.title || row.task}</strong></td><td><span className={`pill ${isGoodHistoryStatus(row.status) ? 'good' : isWarningHistoryStatus(row.status) ? 'warning' : row.status === '失败' ? 'bad' : ''}`}>{row.status}</span></td><td title={row.config} className="truncate">{row.config}</td><td>{row.result_dir ? <button className="link-button" onClick={() => { setResultRoot(row.result_dir!); setPage('results') }}>查看结果 <ArrowRight size={15} /></button> : '—'}</td></tr>)}</tbody></table>{!historyRows.length && <div className="empty-state">暂无任务记录。</div>}</div></div></>}

        {page === 'settings' && settingsDraft && <><div className="page-heading"><div className="eyebrow">PREFERENCES</div><h1>设置</h1><p>管理结果目录、任务记录和 Python 环境。</p></div><div className="card form-card settings-card"><div className="card-heading"><span className="step-number"><Settings2 size={19} /></span><div><h2>工作台设置</h2><p>修改后保存，下一次扫描或运行会使用新值。</p></div></div><div className="form-grid"><label className="form-field wide"><span>默认结果目录</span><PathField value={settingsDraft.results_root} onChange={value => setSettingsDraft({ ...settingsDraft, results_root: value })} kind="dir" /></label><label className="form-field"><span>保留任务记录数</span><input type="number" min={1} max={200} value={settingsDraft.history_limit} onChange={event => setSettingsDraft({ ...settingsDraft, history_limit: Number(event.target.value) })} /></label><label className="form-field wide"><span>任务使用的 Python 解释器</span><PathField value={settingsDraft.python_executable} onChange={value => setSettingsDraft({ ...settingsDraft, python_executable: value })} kind="file" /></label></div><div className="settings-actions"><button className="button secondary danger" onClick={async () => { if (!window.confirm('确定清空任务记录吗？')) return; try { await api('history/clear', {}); setHistoryRows([]); setNotice('任务记录已清空') } catch (err) { setNotice(String(err)) } }}>清空任务记录</button><button className="button primary" onClick={saveSettings}>保存设置</button></div></div><div className="card form-card settings-card"><div className="card-heading"><span className="step-number"><RefreshCw size={19} /></span><div><h2>软件更新</h2><p>当前版本 {version || '—'}。自动检查 GitHub Releases 中的新版本。</p></div></div><p>{updateBusy ? '正在检查或安装更新…' : updateInstalled ? '更新已安装，请关闭并重新打开工作台。' : updateInfo?.message || '正在检查更新…'}</p>{updateError && <p className="inline-error">{updateError}</p>}<div className="settings-actions"><button className="button secondary" onClick={checkForUpdates} disabled={updateBusy}>检查更新</button>{updateInfo?.status === 'available' && !updateInstalled && <button className="button primary" onClick={installUpdate} disabled={updateBusy || active(run)}>下载并安装 {updateInfo.latest_version}</button>}</div></div></>}
      </main>

      <div className={`task-console ${logOpen ? 'expanded' : ''}`}><div className="console-bar"><div className="console-state"><span className={`state-indicator ${run?.status || 'idle'}`} /> <strong>{run ? statusText[run.status] || run.status : '就绪'}</strong><span>{run ? `${catalog[run.tool_id]?.title || run.tool_id} · ${run.started_at}` : '选择工具开始新的任务'}</span></div><div className="console-actions">{active(run) && <button className="button stop" onClick={stop}><Square size={13} fill="currentColor" /> 停止任务</button>}<button className="console-toggle" onClick={() => setLogOpen(!logOpen)}><TerminalSquare size={17} /> 运行详情 <ChevronDown size={15} /></button></div></div>{logOpen && <div className="console-body"><div className="console-body-head"><span>实时日志</span><button onClick={() => navigator.clipboard.writeText(log)} title="复制日志"><Copy size={15} /></button></div><pre ref={logRef}>{log || (active(run) ? '等待任务输出…' : '运行日志会显示在这里。')}</pre></div>}</div>
    </div>
    {notice && <div className="toast" role="status"><AlertCircle size={17} /><span>{notice}</span><button onClick={() => setNotice('')}><X size={15} /></button></div>}
    {configEditor && tool && <ConfigEditor path={configPath} tool={tool} onClose={() => setConfigEditor(false)} onSaved={path => { setConfigPath(path); api<ConfigDocument>(`config?path=${q(path)}`).then(data => setConfigData(data.data)); setNotice('配置已保存') }} />}
  </div>
}

export default App
