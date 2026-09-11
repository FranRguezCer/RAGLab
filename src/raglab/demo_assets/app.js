let sessionToken = '';
const fragment = new URLSearchParams(window.location.hash.slice(1));
sessionToken = fragment.get('token') || '';
history.replaceState(null, '', `${window.location.pathname}${window.location.search}`);

const picker = document.querySelector('#case-picker');
const detail = document.querySelector('#case-detail');
const liveResult = document.querySelector('#live-result');
const collection = document.querySelector('#collection');
const suggestion = document.querySelector('#suggestion');
const query = document.querySelector('#query');
const suggestions = {
  'rpi-computers': ['How do I configure a headless Raspberry Pi?', 'Which Raspberry Pi OS edition fits a server?'],
  'rpi-microcontrollers': ['How do I install MicroPython on a Pico?', 'How can one Pico debug another?'],
  'rpi-camera-ai': ['Where does inference run on the AI Camera?', 'How do I start with Picamera2?'],
};

const escaped = value => String(value ?? '').replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
const prettyName = value => String(value).replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
const metricValue = value => typeof value === 'number' ? `${Math.round(value * 100)}%` : escaped(value);

async function json(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
  return payload;
}

function sourceMarkup(source) {
  const citation = source.citation || {};
  const name = citation.title || citation.source_name || source.document_id || source.id;
  const uri = citation.source_uri || '';
  return `<li>${/^https:\/\//.test(uri) ? `<a href="${escaped(uri)}" target="_blank" rel="noreferrer">${escaped(name)}</a>` : escaped(name)}<br><small>${escaped(citation.heading_path?.join(' › ') || source.retrieval_result_id || '')}</small></li>`;
}

function answerMarkup(item, evaluated) {
  const sources = (item.citations || item.sources || []).length ? `<ul class="source-list">${(item.citations || item.sources).map(sourceMarkup).join('')}</ul>` : '<p class="loading">No sources were cited because the system abstained.</p>';
  const retrieval = item.retrieval || {};
  const generation = item.generation || {metrics: item.metrics};
  const label = evaluated ? 'Evaluated result with ground truth' : 'Live result · no ground truth';
  const duration = evaluated ? `${item.evaluation_duration_seconds}s evaluation run` : `${item.telemetry.duration_ms}ms live query`;
  return `<p class="answer-label">${label}</p><h3>${escaped(item.question || query.value)}</h3><p class="answer-text">${escaped(item.answer)}</p>${sources}<p class="duration">${escaped(duration)}</p><details><summary>Inspect ranking, validation, calls, and tokens</summary><div class="trace-grid"><pre>${escaped(JSON.stringify(retrieval, null, 2))}</pre><pre>${escaped(JSON.stringify(generation, null, 2))}</pre></div></details>`;
}

async function showCase(caseId, button) {
  for (const item of picker.children) item.setAttribute('aria-selected', String(item === button));
  detail.innerHTML = '<p class="loading">Loading evaluated trace…</p>';
  try { detail.innerHTML = answerMarkup(await json(`/v1/demo/cases/${encodeURIComponent(caseId)}`), true); }
  catch (error) { detail.innerHTML = `<p class="error">${escaped(error.message)}</p>`; }
}

function renderStatus(status) {
  document.querySelector('#build-badge').textContent = `commit ${status.provenance.commit.slice(0, 12)}`;
  const metrics = {...status.scorecard.retrieval, ...status.scorecard.generation};
  document.querySelector('#scorecard').innerHTML = Object.entries(metrics).map(([name, value]) => `<article class="metric"><strong>${metricValue(value)}</strong><span>${escaped(prettyName(name))}</span></article>`).join('');
  document.querySelector('#provenance').innerHTML = `<dt>Dataset</dt><dd>${escaped(status.dataset.id)} · ${escaped(status.dataset.sha256)}</dd><dt>Models</dt><dd>${escaped(JSON.stringify(status.models))}</dd><dt>Commit</dt><dd>${escaped(status.provenance.commit)}</dd><dt>Evaluated</dt><dd>${escaped(status.provenance.evaluated_at)} · ${escaped(status.provenance.duration_seconds)}s</dd><dt>Evidence</dt><dd>sha256:${escaped(status.provenance.integrity_sha256)}</dd>`;
  for (const name of Object.keys(status.collections)) collection.add(new Option(prettyName(name), name));
  chooseCollection();
}

function chooseCollection() {
  suggestion.replaceChildren(...suggestions[collection.value].map(value => new Option(value, value)));
  query.value = suggestion.value;
}

collection.addEventListener('change', chooseCollection);
suggestion.addEventListener('change', () => { query.value = suggestion.value; });
document.querySelector('#query-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (!sessionToken) { liveResult.innerHTML = '<p class="error">This page has no active session token. Open the complete share URL again.</p>'; return; }
  liveResult.innerHTML = '<p class="loading">Running retrieval and generation on the local GPU…</p>';
  try {
    const result = await json('/v1/query', {method:'POST', headers:{'content-type':'application/json', authorization:`Bearer ${sessionToken}`}, body:JSON.stringify({query:query.value, collection:collection.value, history:[]})});
    liveResult.innerHTML = answerMarkup(result, false);
  } catch (error) { liveResult.innerHTML = `<p class="error">${escaped(error.message)}</p>`; }
});

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
  } catch (error) { detail.innerHTML = `<p class="error">Prepared evidence unavailable: ${escaped(error.message)}</p>`; }
}

start();
