const $ = selector => document.querySelector(selector);
const tokenKey = 'ragdocument-token';
let documents = [];
let selected = new Set();
let busy = false;
let working = false;
let maxMb = 20;
$('#token').value = localStorage.getItem(tokenKey) || '';

function headers() {
  const token = localStorage.getItem(tokenKey);
  return token ? {Authorization: `Bearer ${token}`} : {};
}

async function api(path, options = {}) {
  const response = await fetch(path, {...options, headers: {...headers(), ...options.headers}});
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Проверьте введённые данные.');
  return data;
}

function notice(text) {
  $('#notice').textContent = text;
  $('#notice').hidden = !text;
}

async function refresh() {
  try {
    const status = await api('/api/status');
    maxMb = status.max_upload_mb;
    $('#formats').textContent = `PDF, DOCX, TXT, MD, XLSX, XLS · до ${maxMb} МБ`;
    $('#connection').textContent = 'Qdrant · Docker';
    if (!status.ollama_available) notice('Ollama недоступна. Проверьте сервис: docker-compose logs ollama');
    else if (status.missing_models.length) notice('Не загружены модели: ' + status.missing_models.join(', ') + '. Запустите подготовку: docker-compose run --rm ollama-init');
    else notice('');
    const previous = new Set(documents.map(doc => doc.id));
    documents = await api('/api/documents');
    for (const doc of documents) if (!previous.has(doc.id)) selected.add(doc.id);
    selected = new Set([...selected].filter(id => documents.some(doc => doc.id === id)));
    renderDocuments();
  } catch (error) { notice(error.message); }
}

function renderDocuments() {
  const container = $('#documents');
  container.replaceChildren();
  $('#count').textContent = documents.length;
  $('#select-all').checked = documents.length > 0 && selected.size === documents.length;
  $('#select-all').indeterminate = selected.size > 0 && selected.size < documents.length;
  if (!documents.length) {
    const empty = document.createElement('p');
    empty.className = 'empty';
    empty.textContent = 'Пока здесь пусто. Добавьте первый документ, чтобы начать.';
    container.append(empty);
  }
  for (const doc of documents) {
    const row = document.createElement('div'); row.className = 'doc';
    const checkbox = document.createElement('input'); checkbox.type = 'checkbox';
    checkbox.checked = selected.has(doc.id); checkbox.setAttribute('aria-label', `Искать в ${doc.filename}`);
    checkbox.addEventListener('change', () => {checkbox.checked ? selected.add(doc.id) : selected.delete(doc.id); renderDocuments();});
    const info = document.createElement('div'); info.className = 'doc-info';
    const name = document.createElement('span'); name.className = 'doc-name'; name.textContent = doc.filename;
    const meta = document.createElement('small'); meta.textContent = `${doc.chunks} фрагм. · ${Math.max(1, Math.round(doc.size / 1024))} КБ`;
    info.append(name, meta);
    const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'delete'; remove.textContent = '×'; remove.setAttribute('aria-label', `Удалить ${doc.filename}`);
    remove.disabled = busy || working;
    remove.addEventListener('click', async () => {
      if (!confirm(`Удалить «${doc.filename}» и его поисковый индекс?`)) return;
      remove.disabled = true;
      try {await api(`/api/documents/${doc.id}`, {method:'DELETE'}); selected.delete(doc.id); await refresh();}
      catch(error) {notice(error.message); remove.disabled = false;}
    });
    row.append(checkbox, info, remove); container.append(row);
  }
}

async function upload(files) {
  if (working || busy) return;
  working = true; $('#files').disabled = true; $('#send').disabled = true; renderDocuments();
  const messages = [];
  for (const file of files) {
    $('#upload-status').textContent = `Индексация: ${file.name}…`;
    try {
      if (file.size > maxMb * 1024 * 1024) throw new Error(`Максимум ${maxMb} МБ`);
      const form = new FormData(); form.append('file', file);
      const doc = await api('/api/documents', {method:'POST', body:form});
      selected.add(doc.id);
      messages.push(`${file.name}: ${doc.duplicate ? 'уже загружен' : 'готово'}`);
    } catch(error) {messages.push(`${file.name}: ${error.message}`);}
  }
  $('#upload-status').textContent = messages.join('\n');
  working = false; $('#files').disabled = false; $('#files').value = ''; $('#send').disabled = false;
  await refresh();
}

function message(role, text) {
  $('#welcome')?.remove();
  const node = document.createElement('div'); node.className = `message ${role}`;
  const title = document.createElement('div'); title.className = 'role'; title.textContent = role === 'user' ? 'ВЫ' : 'ПОМОЩНИК';
  const bubble = document.createElement('div'); bubble.className = 'bubble'; bubble.textContent = text;
  node.append(title, bubble); $('#chat').append(node); node.scrollIntoView({block:'nearest'});
  return {node, bubble};
}

function renderAnswer(target, result) {
  const byId = new Map(result.sources.map(source => [source.id, source]));
  target.bubble.replaceChildren();
  // All model text is inserted as text nodes; only server-issued source URLs become links.
  for (const part of result.answer.split(/(\[\d+\])/g)) {
    const match = part.match(/^\[(\d+)\]$/);
    const source = match && byId.get(Number(match[1]));
    if (source) {
      const link = document.createElement('a'); link.href = source.url; link.target = '_blank'; link.rel = 'noopener'; link.className = 'citation'; link.textContent = part; target.bubble.append(link);
    } else target.bubble.append(document.createTextNode(part));
  }
  if (result.sources.length) {
    const list = document.createElement('div'); list.className = 'sources';
    for (const source of result.sources) {
      const link = document.createElement('a'); link.className = 'source'; link.href = source.url; link.target = '_blank'; link.rel = 'noopener';
      const title = document.createElement('strong'); title.textContent = `[${source.id}] ${source.filename} · ${source.location}`;
      const quote = document.createElement('blockquote'); quote.textContent = source.quote.length > 280 ? source.quote.slice(0,280) + '…' : source.quote;
      link.append(title, quote); list.append(link);
    }
    target.node.append(list);
  }
  target.node.scrollIntoView({block:'nearest'});
}

$('#ask-form').addEventListener('submit', async event => {
  event.preventDefault();
  const question = $('#question').value.trim();
  if (!question || busy || working) return;
  if (!selected.size) {notice('Загрузите и выберите хотя бы один документ.'); return;}
  busy = true; $('#send').disabled = true; $('#send').textContent = 'Поиск…'; renderDocuments();
  const ids = [...selected];
  message('user', question); $('#question').value = '';
  const target = message('assistant', 'Ищу подходящие фрагменты и готовлю ответ…');
  try {renderAnswer(target, await api('/api/ask', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question,document_ids:ids})}));}
  catch(error) {target.bubble.textContent = error.message; target.bubble.classList.add('error');}
  finally {busy = false; $('#send').disabled = false; $('#send').textContent = 'Спросить ↑'; renderDocuments(); $('#question').focus();}
});

$('#question').addEventListener('keydown', event => {if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {event.preventDefault(); $('#ask-form').requestSubmit();}});
$('#select-all').addEventListener('change', event => {selected = new Set(event.target.checked ? documents.map(doc=>doc.id) : []); renderDocuments();});
$('#files').addEventListener('change', event => upload([...event.target.files]));
$('#connect').addEventListener('click', () => {const token = $('#token').value.trim(); token ? localStorage.setItem(tokenKey, token) : localStorage.removeItem(tokenKey); refresh();});
$('#refresh').addEventListener('click', refresh);
$('#clear').addEventListener('click', () => {if (!busy) {$('#chat').replaceChildren(); $('#question').focus();}});
const dropzone = $('#dropzone');
dropzone.addEventListener('keydown', event => {if (event.key === 'Enter' || event.key === ' ') {event.preventDefault(); $('#files').click();}});
dropzone.addEventListener('dragover', event => {event.preventDefault(); dropzone.classList.add('drag');});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
dropzone.addEventListener('drop', event => {event.preventDefault(); dropzone.classList.remove('drag'); upload([...event.dataTransfer.files]);});
refresh();
