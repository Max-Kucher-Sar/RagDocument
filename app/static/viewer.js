(async () => {
  const params = new URLSearchParams(location.search);
  const docId = encodeURIComponent(params.get('document_id') || '');
  const chunk = Number(params.get('chunk') || 0);
  const token = localStorage.getItem('ragdocument-token');
  const headers = token ? {Authorization:`Bearer ${token}`} : {};
  let blobUrl;
  try {
    const response = await fetch(`/api/documents/${docId}/chunks/${chunk}`, {headers});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Не удалось загрузить источник.');
    document.title = data.document.filename + ' — источник';
    document.querySelector('#title').textContent = data.document.filename;
    document.querySelector('#location').textContent = data.chunk.location + ' · выделен фрагмент, использованный для поиска';
    const content = document.querySelector('#content');
    const text = data.unit.text;
    const mark = document.createElement('mark'); mark.textContent = text.slice(data.chunk.start, data.chunk.end);
    content.append(document.createTextNode(text.slice(0, data.chunk.start)), mark, document.createTextNode(text.slice(data.chunk.end)));
    mark.scrollIntoView({block:'center'});
    const button = document.createElement('button'); button.textContent = data.chunk.page ? `Открыть PDF на странице ${data.chunk.page}` : 'Скачать оригинал';
    document.querySelector('#original').append(button);
    button.addEventListener('click', async () => {
      button.disabled = true;
      try {
        const file = await fetch(data.original_url, {headers});
        if (!file.ok) throw new Error('Не удалось загрузить оригинал. Возможно, документ удалён.');
        if (blobUrl) URL.revokeObjectURL(blobUrl);
        blobUrl = URL.createObjectURL(await file.blob());
        if (data.chunk.page) {
          const frame = document.createElement('iframe'); frame.title = 'Оригинальный PDF'; frame.src = `${blobUrl}#page=${data.chunk.page}`;
          document.querySelector('#pdf').replaceChildren(frame); frame.scrollIntoView({block:'start'});
        } else {
          const link = document.createElement('a'); link.href = blobUrl; link.download = data.document.filename; link.click();
        }
      } catch(error) {button.textContent = error.message;}
      finally {button.disabled = false;}
    });
  } catch(error) {document.querySelector('#title').textContent = 'Источник недоступен'; document.querySelector('#content').textContent = error.message;}
  window.addEventListener('beforeunload', () => {if (blobUrl) URL.revokeObjectURL(blobUrl);});
})();
