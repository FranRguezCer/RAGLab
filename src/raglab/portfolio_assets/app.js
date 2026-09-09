const picker = document.querySelector('#case-picker');
const detail = document.querySelector('#case-detail');

const escaped = value => String(value ?? '').replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
const prettyName = value => String(value).replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
const metricValue = value => typeof value === 'number' ? `${Math.round(value * 100)}%` : escaped(value);

async function json(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error((await response.json()).detail || `Request failed (${response.status})`);
  return response.json();
}

function sourceMarkup(source) {
  const citation = source.citation || {};
  const name = citation.title || citation.source_name || source.document_id || source.id;
  const uri = citation.source_uri || '';
  const safeName = escaped(name);
  return `<li>${/^https:\/\//.test(uri) ? `<a href="${escaped(uri)}" target="_blank" rel="noreferrer">${safeName}</a>` : safeName}<br><small>${escaped(citation.heading_path?.join(' › ') || source.retrieval_result_id || '')}</small></li>`;
}

async function showCase(caseId, button) {
  for (const item of picker.children) item.setAttribute('aria-selected', String(item === button));
  detail.innerHTML = '<p class="loading">Loading verified trace…</p>';
  try {
    const item = await json(`/v1/demo/cases/${encodeURIComponent(caseId)}`);
    const sources = item.citations.length ? `<ul class="source-list">${item.citations.map(sourceMarkup).join('')}</ul>` : '<p class="loading">No sources were cited because the system abstained.</p>';
    detail.innerHTML = `<p class="answer-label">${item.abstained ? 'Verified abstention' : 'Verified answer'}</p><h3>${escaped(item.question)}</h3><p class="answer-text">${escaped(item.answer)}</p>${sources}<details><summary>Inspect retrieval and validation trace</summary><div class="trace-grid"><pre>${escaped(JSON.stringify(item.retrieval, null, 2))}</pre><pre>${escaped(JSON.stringify(item.generation, null, 2))}</pre></div></details>`;
  } catch (error) { detail.innerHTML = `<p class="error">${escaped(error.message)}</p>`; }
}

function renderStatus(status) {
  document.querySelector('#build-badge').textContent = `build ${status.release.build_sha.slice(0, 12)}`;
  const metrics = {...status.scorecard.retrieval, ...status.scorecard.generation};
  document.querySelector('#scorecard').innerHTML = Object.entries(metrics).map(([name, value]) => `<article class="metric"><strong>${metricValue(value)}</strong><span>${escaped(prettyName(name))}</span></article>`).join('');
  document.querySelector('#provenance').innerHTML = `<dt>Image</dt><dd>${escaped(status.release.image_digest)}</dd><dt>Dataset</dt><dd>${escaped(status.dataset.id)} · ${escaped(status.dataset.sha256)}</dd><dt>Models</dt><dd>${escaped(JSON.stringify(status.models))}</dd><dt>Produced by</dt><dd><a href="${escaped(status.release.workflow_url)}" target="_blank" rel="noreferrer">GitHub Actions workflow</a> · ${escaped(status.release.created_at)}</dd>`;
}

async function start() {
  try {
    const [status, payload] = await Promise.all([json('/v1/demo/status'), json('/v1/demo/cases')]);
    renderStatus(status);
    payload.cases.forEach((item, index) => {
      const button = document.createElement('button');
      button.className = 'case-button'; button.type = 'button'; button.setAttribute('role', 'listitem');
      button.innerHTML = `<small>${escaped(item.outcome)}</small>${escaped(item.question)}`;
      button.onclick = () => showCase(item.id, button); picker.append(button);
      if (index === 0) showCase(item.id, button);
    });
  } catch (error) { detail.innerHTML = `<p class="error">Verified release unavailable: ${escaped(error.message)}</p>`; }
}

start();
