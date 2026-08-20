const $ = (selector) => document.querySelector(selector);
const state = { files: [], sources: [], jobs: [], activeId: null, activeJob: null, polling: null, mediaDuration: null, cuePage: 1, setupReady: false, expandedProjects: new Set(), cueFilter: 'all', cueSearch: '' };
const videoExt = /\.(mkv|mp4|mov|avi|webm|m4v|ts|mts|m2ts|mpeg|mpg|wmv|mxf|vob|3gp)$/i;
const audioExt = /\.(wav|flac|mp3|m4a|aac|ogg|opus|wma|aiff|aif)$/i;
const subExt = /\.(srt|ass|ssa|vtt|sub|idx)$/i;
const chunkSize = 16 * 1024 * 1024;

document.addEventListener('DOMContentLoaded', () => {
  bindUI(); loadSystem(); loadJobs();
  state.polling = setInterval(loadJobs, 2500);
});

function bindUI() {
  const drop = $('#dropzone'), input = $('#files');
  $('#chooseFiles').onclick = (event) => { event.stopPropagation(); input.click(); };
  drop.onclick = () => input.click();
  drop.onkeydown = (event) => { if (event.key === 'Enter' || event.key === ' ') input.click(); };
  input.onchange = () => selectFiles([...input.files]);
  for (const name of ['dragenter', 'dragover']) drop.addEventListener(name, event => { event.preventDefault(); drop.classList.add('dragging'); });
  for (const name of ['dragleave', 'drop']) drop.addEventListener(name, event => { event.preventDefault(); drop.classList.remove('dragging'); });
  drop.addEventListener('drop', event => selectFiles([...event.dataTransfer.files]));
  $('#startDub').onclick = uploadAndStart;
  $('#startLocal').onclick = startLocal;
  $('#probeLocal').onclick = async () => { try { await probeLocal(); } catch (error) { toast(error.message); } };
  $('#fullRange').onclick = () => { $('#rangeStart').value = ''; $('#rangeEnd').value = ''; updateRangeCopy(); };
  $('#rangeStart').oninput = updateRangeCopy;
  $('#rangeEnd').oninput = updateRangeCopy;
  // The rights confirmation gates the primary action in the open, so the
  // button always states what is missing instead of failing into a toast.
  $('#voiceRights').onchange = updateStartState;
  $('#localPath').oninput = updateStartState;
  $('#cueSearch').oninput = event => { state.cueSearch = event.target.value.trim().toLowerCase(); state.cuePage = 1; renderActiveCues(); };
  $('#cueFilters').onclick = event => {
    const button = event.target.closest('button[data-filter]');
    if (!button) return;
    state.cueFilter = button.dataset.filter; state.cuePage = 1;
    $('#cueFilters').querySelectorAll('button').forEach(item => item.classList.toggle('active', item === button));
    renderActiveCues();
  };
  $('#newJob').onclick = showCreate;
  $('#backButton').onclick = showCreate;
  $('#mobileJobs').onclick = () => $('#rail').classList.toggle('open');
  $('#pauseButton').onclick = () => control('pause');
  $('#cancelButton').onclick = () => control('cancel');
  $('#resumeButton').onclick = () => control('resume');
  $('#approveButton').onclick = approvePending;
  $('#identifyButton').onclick = openMediaMatch;
  $('#mediaMatchClose').onclick = closeMediaMatch;
  $('#mediaSearchButton').onclick = searchMedia;
  $('#mediaSearchQuery').onkeydown = event => { if (event.key === 'Enter') searchMedia(); };
  $('#mediaMatch').onclick = event => { if (event.target === $('#mediaMatch')) closeMediaMatch(); };
  document.addEventListener('keydown', event => { if (event.key === 'Escape' && !$('#mediaMatch').classList.contains('hidden')) closeMediaMatch(); });
}

async function loadSystem() {
  try {
    const info = await api('/api/system');
    const separationReady = info.separator_ready && info.roformer_ready && info.recovery_ready;
    const intelligenceReady = info.asr_ready && info.asr_escalation_ready && info.aligner_ready
      && info.adapter_ready && info.translation_qc_ready && info.visual_speaker_ready && info.tts_fallback_ready;
    const ready = info.ffmpeg && info.ffprobe && info.cuda && info.model_ready
      && info.whisper_ready && intelligenceReady && separationReady;
    state.setupReady = ready;
    const safety = info.gpu_safety || {};
    const unsafe = ['unsafe', 'canary'].includes(safety.status);
    const label = unsafe ? `${info.gpu} · recovering`
      : safety.status === 'active' ? `${info.gpu} · working`
      : ready ? `${info.gpu} · ready` : `${info.gpu} · setup incomplete`;
    $('#systemState').classList.toggle('warning', !ready || unsafe);
    $('#systemState').querySelector('span').textContent = label;
  } catch { $('#systemState').querySelector('span').textContent = 'Engine offline'; }
  updateStartState();
}

// Single source of truth for whether a dub can start, and why not.
function updateStartState() {
  const button = $('#startDub');
  const note = $('#startNote');
  if (!button || !note) return;
  const hasSource = state.sources.length > 0;
  const rights = $('#voiceRights').checked;
  let blocker = null;
  if (!state.setupReady) blocker = 'Finish setup before starting a dub.';
  else if (!hasSource) blocker = 'Choose a film to begin.';
  else if (!rights) blocker = 'Confirm the rights above to continue.';
  button.disabled = Boolean(blocker);
  note.textContent = blocker || (state.sources.length > 1 ? `${state.sources.length} files queued as separate dubs.` : 'Runs unattended to completion.');
  const local = $('#startLocal');
  if (local) local.disabled = !state.setupReady || !rights || !$('#localPath').value.trim();
}

function selectFiles(files) {
  const accepted = files.filter(file => videoExt.test(file.name) || audioExt.test(file.name) || subExt.test(file.name));
  const videos = accepted.filter(file => videoExt.test(file.name) || audioExt.test(file.name) || file.type.startsWith('video/') || file.type.startsWith('audio/'));
  if (!videos.length) return toast('That drop contained no video or audio file.');
  state.files = accepted;
  state.sources = videos;
  const video = videos[0], extras = accepted.filter(file => !videos.includes(file));
  $('#selection').classList.remove('hidden');
  $('#selection').innerHTML = `<div class="file-icon">&#9654;</div><div><strong>${videos.length === 1 ? escapeHtml(video.name) : `${videos.length} files, dubbed separately`}</strong><span>${formatBytes(videos.reduce((sum,item)=>sum+item.size,0))} · ${extras.length ? extras.map(x => escapeHtml(x.name)).join(', ') : 'Embedded subtitles will be checked'}</span></div><button type="button" aria-label="Clear selection">&times;</button>`;
  $('#selection button').onclick = () => { state.files = []; state.sources = []; state.mediaDuration = null; $('#selection').classList.add('hidden'); updateRangeCopy(); updateStartState(); };
  updateStartState();
  inspectBrowserDuration(video);
}

function options() {
  const glossary = Object.fromEntries($('#glossary').value.split(/\r?\n/).map(line => line.split(/\s*=\s*/,2)).filter(parts => parts.length === 2 && parts[0] && parts[1]));
  return { source_language: 'auto', target_language: 'English', subtitle_mode: 'auto',
    audio_mode: 'separate', engine: 'indextts', emotion_mode: 'auto',
    workflow_mode: $('#approvalWorkflow').checked ? 'approval' : 'automatic', mastering_preset: 'cinema',
    range_start: parseClock($('#rangeStart').value), range_end: parseClock($('#rangeEnd').value),
    audio_stream_index: $('#audioTrack').value === '' ? null : Number($('#audioTrack').value), glossary,
    subtitle_stream_index: $('#subtitleTrack').value === '' ? null : Number($('#subtitleTrack').value),
    voice_rights_confirmed: $('#voiceRights').checked };
}

function validateRange() {
  if (!$('#voiceRights').checked) throw new Error('Confirm the rights before starting a dub.');
  const start = parseClock($('#rangeStart').value), end = parseClock($('#rangeEnd').value);
  if ($('#rangeStart').value.trim() && start === null) throw new Error('Start must be MM:SS or HH:MM:SS.');
  if ($('#rangeEnd').value.trim() && end === null) throw new Error('End must be MM:SS or HH:MM:SS.');
  if (start !== null && end !== null && end <= start) throw new Error('The end must come after the start.');
  if (state.mediaDuration && start !== null && start >= state.mediaDuration) throw new Error('The start is past the end of the film.');
  if (state.mediaDuration && end !== null && end > state.mediaDuration + .25) throw new Error('The end is past the length of the film.');
}

async function uploadAndStart() {
  const button = $('#startDub'); button.disabled = true; $('#startNote').textContent = 'Uploading…';
  try {
    if (!state.setupReady) throw new Error('Finish setup before starting a dub.');
    validateRange();
    let job;
    for (const source of state.sources) {
      const stem=source.name.replace(/\.[^.]+$/,'').toLowerCase();
      const matching=state.files.filter(file=>subExt.test(file.name) && (state.sources.length===1 || file.name.replace(/\.[^.]+$/,'').toLowerCase()===stem));
      const jobFiles=[source,...matching];
      const specs = jobFiles.map(file => ({ name: file.name, size: file.size, kind: kindFor(file) }));
      job = await api('/api/jobs', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({files: specs, options: options()}) });
      state.activeId = job.id; showJob(job);
      for (let index = 0; index < jobFiles.length; index++) {
        const file = jobFiles[index], upload = job.uploads[index]; let offset = upload.received || 0;
        while (offset < file.size) {
          const body = file.slice(offset, Math.min(file.size, offset + chunkSize));
          const response = await fetch(`/api/jobs/${job.id}/files/${upload.id}`, { method:'PUT', headers:{'Upload-Offset':String(offset)}, body });
          const data = await response.json();
          if (!response.ok && response.status !== 409) throw new Error(data.detail || 'Upload interrupted');
          offset = data.offset;
          const sent = jobFiles.slice(0,index).reduce((sum,f)=>sum+f.size,0)+offset;
          renderUpload(sent/jobFiles.reduce((sum,f)=>sum+f.size,0)*100,file.name);
        }
      }
      job = await api(`/api/jobs/${job.id}/finalize`, {method:'POST'});
    }
    renderJob(job); await loadJobs(); toast(`${state.sources.length} dub${state.sources.length===1?'':'s'} queued.`);
  } catch (error) { toast(error.message); updateStartState(); }
}

async function startLocal() {
  const paths = $('#localPath').value.split(/\r?\n/).map(value=>value.trim()).filter(Boolean);
  if (!paths.length) return toast('Enter at least one full file path.');
  try {
    if (!state.setupReady) throw new Error('Finish setup before starting a dub.');
    await probeLocal(); validateRange();
    let job;
    for (const path of paths) {
      const jobOptions = {...options()};
      // The visible selector describes the first probed file.  Other files in a
      // batch are independently auto-selected so stream indexes are never
      // accidentally copied across unrelated containers.
      if (paths.length > 1) { jobOptions.audio_stream_index = null; jobOptions.subtitle_stream_index = null; }
      job = await api('/api/jobs/local', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({path, options:jobOptions})});
    }
    state.activeId = job.id; showJob(job); await loadJobs(); toast(`${paths.length} dub${paths.length === 1 ? '' : 's'} queued.`);
  } catch (error) { toast(error.message); }
}

async function probeLocal() {
  const path = $('#localPath').value.split(/\r?\n/).map(value=>value.trim()).filter(Boolean)[0];
  if (!path) throw new Error('Enter the full path to a file.');
  const info = await api('/api/media/probe', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({path})});
  state.mediaDuration = Number(info.duration);
  const tracks = info.audio_streams || [], select = $('#audioTrack');
  select.innerHTML = tracks.map(track => `<option value="${track.index}">${escapeHtml(`${track.ordinal + 1}. ${track.title} · ${track.language} · ${track.channels}ch ${track.codec}`)}</option>`).join('');
  $('#audioTrackRow').classList.toggle('hidden', tracks.length < 2);
  const preferred = tracks.find(track => track.default && !/commentary|description/i.test(track.title)) || tracks[0];
  if (preferred) select.value = String(preferred.index);
  const subtitles = (info.subtitle_streams || []).filter(track=>track.text), subtitleSelect = $('#subtitleTrack');
  subtitleSelect.innerHTML = `<option value="">Automatic / ASR</option>` + subtitles.map(track => `<option value="${track.index}">${escapeHtml(`${track.title} · ${track.language} · ${track.codec}${track.forced ? ' · forced' : ''}`)}</option>`).join('');
  $('#subtitleTrackLabel').classList.toggle('hidden', subtitles.length < 2);
  updateRangeCopy(); return info;
}

function inspectBrowserDuration(file) {
  state.mediaDuration = null; updateRangeCopy();
  const video = document.createElement('video'), url = URL.createObjectURL(file);
  video.preload = 'metadata';
  video.onloadedmetadata = () => {
    state.mediaDuration = Number.isFinite(video.duration) ? video.duration : null;
    URL.revokeObjectURL(url); updateRangeCopy();
  };
  video.onerror = () => URL.revokeObjectURL(url);
  video.src = url;
}

function updateRangeCopy() {
  const start = parseClock($('#rangeStart').value), end = parseClock($('#rangeEnd').value);
  if (start !== null || end !== null) {
    const finish = end !== null ? formatCueTime(end) : (state.mediaDuration ? formatCueTime(state.mediaDuration) : 'end');
    $('#sourceDuration').textContent = `Dubbing ${formatCueTime(start || 0)} to ${finish}. Speaker registration still scans the whole film.`;
  } else {
    $('#sourceDuration').textContent = state.mediaDuration ? `${formatCueTime(state.mediaDuration)} detected. Whole film selected.` : 'Leave blank to dub the whole film.';
  }
}

function kindFor(file) {
  if (/\.idx$/i.test(file.name)) return 'subtitle_index';
  if (subExt.test(file.name)) return 'subtitle';
  if (audioExt.test(file.name) || file.type.startsWith('audio/')) return 'audio';
  return 'video';
}

async function loadJobs() {
  try {
    state.jobs = await api('/api/jobs'); renderJobList();
    if (state.activeId) {
      const summary = state.jobs.find(job => job.id === state.activeId);
      if (summary) {
        const stale = !state.activeJob || state.activeJob.id !== summary.id
          || state.activeJob.cue_revision !== summary.cue_revision
          || state.activeJob.log_revision !== summary.log_revision;
        if (stale) state.activeJob = await api(`/api/jobs/${summary.id}`);
        else state.activeJob = {...state.activeJob, ...summary, cues:state.activeJob.cues, logs:state.activeJob.logs};
        renderJob(state.activeJob);
      }
    }
  } catch { /* The next poll will recover. */ }
}

function renderJobList() {
  const list = $('#jobList');
  if (!state.jobs.length) { list.innerHTML = '<div class="list-empty">No projects yet.<br>Drop a film to start one.</div>'; return; }
  const projects = new Map();
  for (const job of state.jobs) {
    const key = job.project?.id || job.project_id || `job-${job.id}`;
    if (!projects.has(key)) projects.set(key, {project:job.project || null, jobs:[]});
    projects.get(key).jobs.push(job);
  }
  list.innerHTML = [...projects.values()].map(group => {
    const jobs = group.jobs.sort((a,b)=>Number(b.created_at||0)-Number(a.created_at||0));
    const latest = jobs[0], project = group.project || {};
    const active = jobs.some(job=>job.id===state.activeId), running = jobs.find(job=>['uploading','queued','processing','paused','awaiting_selection'].includes(job.status)) || latest;
    const passed = latest.status === 'complete' && !Number(latest.qc?.flagged_count || 0) && latest.qc?.passed !== false;
    const title = project.title || latest.media_identity?.title || latest.filename;
    const art = project.poster_url ? `<img src="${escapeHtml(project.poster_url)}" alt="" loading="lazy">` : `<span class="cover-letter">${escapeHtml(initials(title).slice(0,1))}</span>`;
    const stateKind = passed ? 'pass'
      : latest.status === 'error' ? 'error'
      : ['complete','needs_review'].includes(latest.status) ? 'review'
      : ['processing','queued','uploading','awaiting_selection'].includes(latest.status) ? 'live' : 'idle';
    const projectKey=project.id||`job-${latest.id}`, expanded=state.expandedProjects.has(projectKey);
    const children = expanded ? `<div class="project-tasks">${jobs.map((job,index)=>`<button class="project-task ${job.id===state.activeId?'active':''} ${['processing','queued','uploading'].includes(job.status)?'running':''}" data-id="${job.id}"><i></i><span>${escapeHtml(taskLabel(job,index,jobs.length))}<small>${escapeHtml(labelStatus(job))} · ${relativeTime(job.updated_at)}</small></span></button>`).join('')}</div>` : '';
    return `<article class="project-item ${active?'active':''} ${expanded?'expanded':''}"><button class="project-main" data-project="${escapeHtml(projectKey)}" aria-expanded="${expanded}"><span class="project-cover">${art}<span class="project-state" data-state="${stateKind}"></span></span><span class="project-copy"><strong>${escapeHtml(title)}</strong><small>${escapeHtml(projectSubtitle(project,latest))}</small><em>${jobs.length} ${jobs.length===1?'task':'tasks'} · ${escapeHtml(labelStatus(running))}</em></span><span class="project-chevron" aria-hidden="true">&#8250;</span><i class="project-progress" style="--p:${Number(running.progress)||0}"></i></button>${children}</article>`;
  }).join('');
  list.querySelectorAll('[data-project]').forEach(button => button.onclick = () => {
    const key=button.dataset.project;
    if(state.expandedProjects.has(key))state.expandedProjects.delete(key);
    else { state.expandedProjects.clear(); state.expandedProjects.add(key); }
    renderJobList();
  });
  list.querySelectorAll('[data-id]').forEach(button => button.onclick = async () => {
    state.activeId = button.dataset.id;
    try { state.activeJob = await api(`/api/jobs/${state.activeId}`); showJob(state.activeJob); }
    catch (error) { toast(error.message); }
    $('#rail').classList.remove('open');
  });
}

function taskLabel(job, index, count) {
  const start=job.options?.range_start, end=job.options?.range_end;
  if (start != null || end != null) return `${formatCueTime(start||0)}–${end==null?'end':formatCueTime(end)}`;
  const created = new Date(Number(job.created_at||0)*1000);
  return count > 1 ? `Whole film · ${created.toLocaleDateString(undefined,{day:'numeric',month:'short'})}` : 'Whole film';
}

function projectSubtitle(project, job) {
  const kind = project.media_type === 'tv' ? 'TV series' : 'Film';
  const year = project.year ? ` · ${project.year}` : '';
  const episode = job.media_identity?.season != null ? ` · S${String(job.media_identity.season).padStart(2,'0')}E${String(job.media_identity.episode).padStart(2,'0')}` : '';
  return `${kind}${year}${episode}`;
}

function showCreate() {
  state.activeId = null; state.activeJob = null; $('#jobView').classList.add('hidden'); $('#createView').classList.remove('hidden'); renderJobList();
}

function showJob(job) {
  state.activeJob = job; $('#createView').classList.add('hidden'); $('#jobView').classList.remove('hidden'); renderJob(job); renderJobList();
}

function renderUpload(percent, filename) {
  const job = { id:state.activeId, filename, status:'uploading', stage:`Uploading ${filename}`, progress:percent, cues:[], logs:['Uploading in resumable 16 MB pieces'] };
  renderJob(job);
}

function renderJob(job) {
  if (!job) return;
  const progress = Math.max(0, Math.min(100, Number(job.progress)||0));
  const complete = ['complete','needs_review'].includes(job.status), failed = job.status === 'error';
  const deliveryFailed = complete && job.qc?.passed === false;
  const cueReview = complete && Number(job.qc?.flagged_count || 0) > 0;
  const project = job.project || {};
  const identity = job.media_identity || {};
  const displayTitle = project.title || identity.title || job.filename;
  $('#jobTitle').textContent = displayTitle;
  $('#jobStatus').textContent = labelStatus(job);
  $('#jobStatus').className = `chip ${chipClassFor(job)}`;
  $('#jobStage').textContent = job.stage || 'Queued';
  $('#jobPercent').textContent = `${Math.round(progress)}%`;
  if (job.status === 'processing' && job.processing_started_at) {
    const elapsed = Number(job.active_processing_seconds||0) + (job.active_run_started_at ? Date.now()/1000-Number(job.active_run_started_at) : 0);
    const predicted=Number(job.eta?.predicted_seconds||0);
    $('#etaLabel').textContent = predicted > 0
      ? `${formatTime(Math.max(0,predicted-elapsed))} left · ${formatTime(elapsed)} elapsed`
      : `${formatTime(elapsed)} elapsed · estimating`;
  } else $('#etaLabel').textContent = ({
    queued: 'Waiting for the GPU worker',
    uploading: 'Uploading',
    paused: 'Paused',
    awaiting_selection: 'Waiting for your track choice',
    cancelled: 'Cancelled',
  })[job.status] || (complete ? 'Finished' : failed ? 'Stopped' : 'Waiting to start');
  $('#progressBar').style.width = `${progress}%`;
  $('#progressTrack').setAttribute('aria-valuenow', String(Math.round(progress)));
  $('#progressRing').classList.toggle('hidden', complete || failed);
  document.querySelector('.stage-line .dot').className = `dot ${failed ? 'dot-error' : complete ? 'dot-pass' : 'dot-live'}`;
  $('#progressRing').style.setProperty('--progress', `${progress * 3.6}deg`);
  const episode = identity.season != null ? `S${String(identity.season).padStart(2,'0')}E${String(identity.episode).padStart(2,'0')}${identity.episode_end?`–E${String(identity.episode_end).padStart(2,'0')}`:''}` : null;
  $('#jobIdentityLine').textContent = [project.media_type==='tv'?'TV series':'Film',project.year||identity.year,episode].filter(Boolean).join(' · ');
  const mediaDetail = job.media ? `${job.media.video_codec?.toUpperCase() || job.media.media_kind?.toUpperCase() || 'MEDIA'} · ${formatTime(job.media.duration)}` : 'Preparing source details';
  $('#jobMeta').textContent = `${mediaDetail} · ${job.filename || 'source media'}`;
  const poster = $('#jobPoster'), posterBox=$('#heroPoster'), fallback=$('#jobPosterFallback');
  fallback.textContent = initials(displayTitle).slice(0,1) || 'D';
  if (project.poster_url) { poster.src=project.poster_url; poster.alt=`${displayTitle} cover`; poster.classList.remove('hidden'); posterBox.classList.add('has-poster'); poster.onerror=()=>{poster.classList.add('hidden');posterBox.classList.remove('has-poster');}; }
  else { poster.removeAttribute('src'); poster.classList.add('hidden'); posterBox.classList.remove('has-poster'); }
  const identityButton=$('#identifyButton');
  identityButton.textContent = identity.needs_confirmation ? 'Confirm the suggested match' : (identity.matched ? 'Change title or cover' : 'Find title and cover');
  identityButton.classList.toggle('attention', Boolean(identity.needs_confirmation));
  const steps = [...document.querySelectorAll('.pipeline-steps span')];
  const reached = steps.filter(step => progress >= Number(step.dataset.at)).length;
  const running = ['processing','queued','uploading'].includes(job.status);
  // The stage in flight is the last one whose threshold progress has passed.
  // A finished job marks every stage done rather than leaving the last one open.
  const active = complete ? steps.length : Math.max(0, reached - 1);
  steps.forEach((step, index) => {
    step.classList.toggle('done', index < active);
    step.classList.toggle('current', running && index === active);
  });
  $('#cueSource').textContent = job.cue_source || 'Detecting';
  $('#lineCount').textContent = (job.cues?.length || job.cue_count) ? (job.cues?.length || job.cue_count).toLocaleString() : '—';
  $('#runtime').textContent = job.media ? formatTime(job.media.duration) : '—';
  $('#dimensions').textContent = job.media?.width ? `${job.media.width} × ${job.media.height}` : '—';
  renderCues(job.cues || [], job.status === 'needs_review' || (complete && job.options?.workflow_mode === 'review'), job.status === 'paused');
  $('#logList').innerHTML = (job.logs || []).map(item => `<p>${escapeHtml(item)}</p>`).join('') || '<p>No notes yet.</p>';
  $('#resultBanner').classList.toggle('hidden', !complete);
  $('#resultBanner').classList.toggle('delivery-failed', deliveryFailed || cueReview);
  $('#errorBanner').classList.toggle('hidden', !failed);
  const awaitingApproval = job.status === 'paused' && job.stage === 'Translation ready for approval';
  const awaitingTrack = job.status === 'awaiting_selection';
  $('#approvalBanner').classList.toggle('hidden', !(awaitingApproval || awaitingTrack));
  $('#approvalTitle').textContent = awaitingTrack ? 'Choose the tracks' : 'Translation ready';
  $('#approvalText').textContent = awaitingTrack
    ? 'Pick the full programme audio, not commentary or description, and a dialogue subtitle track if one helps.'
    : 'Correct any line below. Voicing starts when you continue.';
  $('#jobTrackPickers').classList.toggle('hidden', !awaitingTrack);
  $('#approveButton').textContent = awaitingTrack ? 'Start dub' : 'Continue to voicing';
  if (awaitingTrack) {
    const audio=(job.media_selection?.audio_streams || []), subtitles=(job.media_selection?.subtitle_streams || []);
    $('#jobAudioTrack').innerHTML = audio.map(track=>`<option value="${track.index}">${escapeHtml(`${track.ordinal+1}. ${track.title} · ${track.language} · ${track.channels}ch`)}</option>`).join('');
    const preferredAudio=audio.find(track=>track.default&&!/commentary|description|director|isolated|music only|karaoke/i.test(track.title))
      || audio.find(track=>!/commentary|description|director|isolated|music only|karaoke/i.test(track.title)) || audio[0];
    if(preferredAudio)$('#jobAudioTrack').value=String(preferredAudio.index);
    $('#jobSubtitleTrack').innerHTML = `<option value="">Automatic / ASR</option>`+subtitles.map(track=>`<option value="${track.index}">${escapeHtml(`${track.title} · ${track.language}${track.forced?' · forced':''}`)}</option>`).join('');
    const preferredSubtitle=subtitles.find(track=>!track.forced&&!/commentary|director|signs|songs|trivia/i.test(track.title));
    if(preferredSubtitle)$('#jobSubtitleTrack').value=String(preferredSubtitle.index);
  }
  $('#pauseButton').classList.toggle('hidden', !['processing','queued'].includes(job.status));
  $('#cancelButton').classList.toggle('hidden', !['processing','queued','paused','uploading','awaiting_selection'].includes(job.status));
  if (complete) {
    $('#downloadButton').href = `/api/jobs/${job.id}/download`; $('#qcButton').href = `/api/jobs/${job.id}/qc`;
    $('#srtExport').href = `/api/jobs/${job.id}/export/srt`; $('#csvExport').href = `/api/jobs/${job.id}/export/csv`;
    $('#edlExport').href = `/api/jobs/${job.id}/export/edl`; $('#clipsExport').href = `/api/jobs/${job.id}/export/clips`;
    $('#mixExport').href = `/api/jobs/${job.id}/export/mix`; $('#dialogueExport').href = `/api/jobs/${job.id}/export/dialogue`;
    const flagged = Number(job.qc?.flagged_count || 0);
    const failures = (job.qc?.failures || []).length;
    $('#resultLabel').textContent = (deliveryFailed || cueReview) ? 'NEEDS REVIEW' : 'DUB READY';
    $('#resultTitle').textContent = deliveryFailed ? 'Delivery checks failed.'
      : cueReview ? `${flagged} line${flagged === 1 ? '' : 's'} need review.`
      : 'Every automatic check passed.';
    $('#outputSize').textContent = [
      `${formatBytes(job.output_size)} MKV`,
      flagged ? `${flagged} flagged line${flagged === 1 ? '' : 's'}` : null,
      failures ? `${failures} delivery issue${failures === 1 ? '' : 's'}` : null,
    ].filter(Boolean).join(' · ');
  }
  if (failed) $('#errorText').textContent = job.error || 'The run stopped for an unknown reason.';
}

function renderCues(cues, flaggedOnly = false, editable = false) {
  state.cueSet = { cues, flaggedOnly, editable };
  renderActiveCues();
}

// Filtering and search live here so the toolbar can re-render without a refetch.
function renderActiveCues() {
  const list = $('#cueList');
  if (!list) return;
  const { cues = [], flaggedOnly = false, editable = false } = state.cueSet || {};
  const total = flaggedOnly ? cues.filter(cue => cue.needs_review).length : cues.length;
  let pool = flaggedOnly ? cues.filter(cue => cue.needs_review) : cues;
  if (state.cueFilter === 'review') pool = pool.filter(cue => cue.needs_review);
  else if (state.cueFilter === 'done') pool = pool.filter(cue => !cue.needs_review && ['complete', 'voiced'].includes(cue.status));
  if (state.cueSearch) {
    pool = pool.filter(cue => `${cue.english || ''} ${cue.source || ''} ${cue.speaker || ''}`.toLowerCase().includes(state.cueSearch));
  }

  const count = $('#cueCount');
  if (count) count.textContent = !total ? '' : (pool.length === total ? total.toLocaleString() : `${pool.length.toLocaleString()} of ${total.toLocaleString()}`);

  if (!pool.length) {
    list.innerHTML = `<div class="cue-empty">${total ? 'No lines match this filter.' : 'Lines appear once dialogue analysis finishes.'}</div>`;
    return;
  }

  const visible = pool.slice(0, state.cuePage * 500);
  list.innerHTML = visible.map(cue => {
    const voice = cue.speaker_id == null ? 'Identifying' : (cue.speaker || 'Unassigned');
    const note = cue.needs_review
      ? `Needs review · ${(cue.review_reasons || []).join('; ')}`
      : (cue.performance_source || cue.emotion || 'Analysis pending');
    const actions = editable
      ? `<div class="cue-edit-actions"><button type="button" class="cue-fix" data-cue="${cue.id}" data-text="${escapeHtml(cue.english || '')}">Edit</button><button type="button" class="cue-split" data-cue="${cue.id}">Split</button><button type="button" class="cue-merge" data-cue="${cue.id}">Merge</button></div>`
      : cue.needs_review
        ? `<div class="cue-edit-actions"><button type="button" class="cue-fix" data-cue="${cue.id}" data-text="${escapeHtml(cue.english || '')}">Fix</button><button type="button" class="cue-takes" data-cue="${cue.id}">Takes</button></div>`
        : `<span class="cue-state ${cue.status}">${['complete', 'voiced'].includes(cue.status) ? 'Passed' : 'Waiting'}</span>`;
    return `<div class="cue-row ${cue.needs_review ? 'needs-review' : ''}">`
      + `<time>${formatCueTime(cue.start)}<small>${formatCueTime(cue.end)}</small></time>`
      + `<div class="cue-text"><strong>${escapeHtml(cue.english || '')}</strong><span>${escapeHtml(note)}</span></div>`
      + `<span class="voice-name"><i>${cue.speaker_id == null ? '?' : escapeHtml(initials(voice))}</i><span>${escapeHtml(voice)}</span></span>`
      + actions
      + `</div>`;
  }).join('') + (visible.length < pool.length
    ? `<button type="button" class="cue-more">Show ${Math.min(500, pool.length - visible.length).toLocaleString()} more</button>`
    : '');

  list.querySelectorAll('.cue-fix').forEach(button => button.onclick = () => editAndRegenerate(button));
  list.querySelectorAll('.cue-split').forEach(button => button.onclick = () => splitCue(button.dataset.cue));
  list.querySelectorAll('.cue-merge').forEach(button => button.onclick = () => mergeCue(button.dataset.cue));
  list.querySelectorAll('.cue-takes').forEach(button => button.onclick = () => restoreTake(button.dataset.cue));
  const more = list.querySelector('.cue-more');
  if (more) more.onclick = () => { state.cuePage += 1; renderActiveCues(); };
}

async function openMediaMatch() {
  if (!state.activeJob) return;
  const identity=state.activeJob.media_identity||{}, project=state.activeJob.project||{};
  $('#mediaSearchQuery').value=project.title||identity.title||state.activeJob.filename||'';
  $('#mediaSearchType').value=project.media_type||identity.media_type||'movie';
  $('#mediaSearchYear').value=project.year||identity.year||'';
  $('#mediaMatch').classList.remove('hidden');
  document.body.style.overflow='hidden';
  const suggestions=identity.candidates||[];
  if(suggestions.length) renderMediaResults(suggestions);
  else $('#mediaResults').innerHTML='<div class="media-results-empty">Search by the on-screen title. Adding a year usually pins the result.</div>';
  setTimeout(()=>$('#mediaSearchQuery').focus(),30);
}

function closeMediaMatch() {
  $('#mediaMatch').classList.add('hidden');
  if ($('#setupWizard').classList.contains('hidden')) document.body.style.overflow='';
  $('#identifyButton').focus();
}

async function searchMedia() {
  const query=$('#mediaSearchQuery').value.trim();
  if(query.length<2)return toast('Type at least two characters.');
  const button=$('#mediaSearchButton');button.disabled=true;button.textContent='Searching…';
  $('#mediaResults').innerHTML='<div class="media-results-empty">Searching…</div>';
  try{
    const params=new URLSearchParams({q:query,media_type:$('#mediaSearchType').value});
    const year=$('#mediaSearchYear').value;if(year)params.set('year',year);
    renderMediaResults(await api(`/api/library/search?${params}`));
  }catch(error){
    $('#mediaResults').innerHTML=`<div class="media-results-empty">${escapeHtml(error.message)}<br><br><button type="button" class="btn-secondary" id="openSetupFromMatch">Open Setup</button></div>`;
    const open=$('#openSetupFromMatch');if(open)open.onclick=()=>{closeMediaMatch();$('#setupButton').click();};
  }finally{button.disabled=false;button.textContent='Search';}
}

function renderMediaResults(results) {
  const list=$('#mediaResults');
  if(!results.length){list.innerHTML='<div class="media-results-empty">No close matches. Try the original release title, or drop the year.</div>';return;}
  list.innerHTML=results.map(item=>`<button class="media-result" data-provider="${escapeHtml(item.provider)}" data-type="${escapeHtml(item.media_type)}" data-external-id="${Number(item.id)}"><span><strong>${escapeHtml(item.title)}</strong><small>${escapeHtml([item.year,item.media_type==='tv'?'TV series':'Film',item.overview].filter(Boolean).join(' · '))}</small></span><span>Select</span></button>`).join('');
  list.querySelectorAll('.media-result').forEach(button=>button.onclick=()=>chooseMediaResult(button));
}

async function chooseMediaResult(button) {
  button.disabled=true;const marker=button.lastElementChild;marker.textContent='Saving…';
  try{
    const job=await api(`/api/jobs/${state.activeId}/identify`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({provider:button.dataset.provider,media_type:button.dataset.type,external_id:Number(button.dataset.externalId)})});
    state.activeJob=job;renderJob(job);closeMediaMatch();await loadJobs();toast('Title and cover updated.');
  }catch(error){toast(error.message);button.disabled=false;marker.textContent='Select';}
}

async function editAndRegenerate(button) {
  const cue = (state.activeJob?.cues || []).find(item=>String(item.id)===String(button.dataset.cue));
  const source = prompt('Source transcript', cue?.source || '');
  if (source === null) return;
  const revised = prompt('English dialogue', button.dataset.text);
  if (!revised) return;
  const speaker = prompt('Character name (renames this voice across the project)', cue?.speaker || '');
  if (speaker === null || !speaker.trim()) return;
  const startText = prompt('Line start (MM:SS or HH:MM:SS)', formatCueTime(cue?.start || 0));
  if (startText === null) return;
  const endText = prompt('Line end (MM:SS or HH:MM:SS)', formatCueTime(cue?.end || 0));
  if (endText === null) return;
  const start=parseClock(startText), end=parseClock(endText);
  if(start===null || end===null || end<=start)return toast('Enter a start and an end after it.');
  const unchanged = revised.trim() === String(cue?.english || '').trim()
    && source.trim() === String(cue?.source || '').trim()
    && speaker.trim() === String(cue?.speaker || '').trim()
    && Math.abs(start-Number(cue?.start||0))<.001 && Math.abs(end-Number(cue?.end||0))<.001;
  if (unchanged) return;
  try {
    const edited = await api(`/api/jobs/${state.activeId}/cues/${button.dataset.cue}`, {method:'PATCH', headers:{'Content-Type':'application/json'}, body:JSON.stringify({source:source.trim(),english:revised.trim(),speaker_name:speaker.trim(),start,end})});
    if (state.activeJob?.stage === 'Translation ready for approval') {
      state.activeJob = edited; renderJob(edited); toast('Translation updated.');
    } else {
      const job = await api(`/api/jobs/${state.activeId}/cues/${button.dataset.cue}/regenerate`, {method:'POST'});
      state.activeJob = job; renderJob(job); toast('Line queued for regeneration.');
    }
  } catch (error) { toast(error.message); }
}

async function splitCue(cueId) {
  const cue=(state.activeJob?.cues||[]).find(item=>String(item.id)===String(cueId)); if(!cue)return;
  const atText=prompt('Split at (MM:SS or HH:MM:SS)',formatCueTime((Number(cue.start)+Number(cue.end))/2)); if(!atText)return;
  const at=parseClock(atText); if(at===null)return toast('Enter a valid time.');
  const first=prompt('First line',cue.english||''); if(!first)return;
  const second=prompt('Second line',''); if(!second)return;
  try{state.activeJob=await api(`/api/jobs/${state.activeId}/cues/${cueId}/split`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({at,first_text:first,second_text:second})});renderJob(state.activeJob);}catch(error){toast(error.message);}
}

async function mergeCue(cueId) {
  if(!confirm('Merge this line into the next one? Both voice takes are regenerated.'))return;
  try{state.activeJob=await api(`/api/jobs/${state.activeId}/cues/${cueId}/merge-next`,{method:'POST'});renderJob(state.activeJob);}catch(error){toast(error.message);}
}

async function restoreTake(cueId) {
  try {
    const takes=await api(`/api/jobs/${state.activeId}/cues/${cueId}/takes`); if(!takes.length)return toast('No earlier takes stored for this line.');
    const choice=prompt(`Take ID:\n${takes.map(t=>`${t.id} · ${t.files.join(', ')}`).join('\n')}`,takes[0].id); if(!choice)return;
    state.activeJob=await api(`/api/jobs/${state.activeId}/cues/${cueId}/takes/${encodeURIComponent(choice)}/restore`,{method:'POST'});renderJob(state.activeJob);toast('Earlier take restored and queued for QC.');
  } catch(error){toast(error.message);}
}

async function control(action) {
  if (!state.activeId) return;
  try { const job = await api(`/api/jobs/${state.activeId}/control/${action}`, {method:'POST'}); state.activeJob = job; renderJob(job); }
  catch (error) { toast(error.message); }
}

async function approvePending() {
  if (!state.activeId) return;
  if (state.activeJob?.status === 'awaiting_selection') {
    try { const subtitle=$('#jobSubtitleTrack').value; const job=await api(`/api/jobs/${state.activeId}/media-tracks`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({audio_index:Number($('#jobAudioTrack').value),subtitle_index:subtitle===''?null:Number(subtitle)})}); state.activeJob=job;renderJob(job); }
    catch(error){toast(error.message);} return;
  }
  control('resume');
}

async function api(url, options) {
  const response = await fetch(url, options); const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'The local service could not complete that request.');
  return data;
}

function labelStatus(job) {
  if (job.status === 'complete' && (Number(job.qc?.flagged_count || 0) || job.qc?.passed === false)) return 'Needs review';
  return ({uploading:'Uploading',awaiting_selection:'Choose tracks',queued:'Queued',processing:'Dubbing',paused:'Paused',cancelled:'Cancelled',complete:'Passed',needs_review:'Needs review',error:'Stopped'})[job.status] || job.status;
}

// Status colour is semantic: amber means working, green passed, terracotta
// needs a human, red stopped. Everything else stays neutral.
function chipClassFor(job) {
  if (job.status === 'error') return 'chip-error';
  if (job.status === 'complete' && !Number(job.qc?.flagged_count || 0) && job.qc?.passed !== false) return 'chip-pass';
  if (['complete', 'needs_review', 'awaiting_selection', 'paused'].includes(job.status)) return 'chip-review';
  if (['processing', 'queued', 'uploading'].includes(job.status)) return 'chip-live';
  return '';
}
function formatBytes(bytes=0) { if (!bytes) return '0 B'; const units=['B','KB','MB','GB','TB'], i=Math.min(units.length-1,Math.floor(Math.log(bytes)/Math.log(1024))); return `${(bytes/1024**i).toFixed(i?1:0)} ${units[i]}`; }
function formatTime(seconds=0) { const h=Math.floor(seconds/3600), m=Math.floor(seconds%3600/60), s=Math.floor(seconds%60); return h ? `${h}h ${m}m` : `${m}:${String(s).padStart(2,'0')}`; }
function formatCueTime(seconds=0) { const h=Math.floor(seconds/3600), m=Math.floor(seconds%3600/60), s=Math.floor(seconds%60); return h ? `${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}` : `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`; }
function parseClock(value='') { const text=String(value).trim(); if(!text)return null; const parts=text.split(':'); if(parts.length>3||parts.some(x=>!/^\d+(?:\.\d+)?$/.test(x)))return null; let seconds=0; for(const part of parts)seconds=seconds*60+Number(part); return Number.isFinite(seconds)&&seconds>=0?seconds:null; }
function relativeTime(ts) { const delta=Math.max(0,Date.now()/1000-ts); if(delta<60)return 'now'; if(delta<3600)return `${Math.floor(delta/60)}m ago`; if(delta<86400)return `${Math.floor(delta/3600)}h ago`; return `${Math.floor(delta/86400)}d ago`; }
function initials(text='Voice') { return text.split(/\s+/).map(x=>x[0]).join('').slice(0,2).toUpperCase(); }
function escapeHtml(value='') { return String(value).replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char])); }
function toast(message) { const el=$('#toast'); el.textContent=message; el.classList.remove('hidden'); clearTimeout(toast.timer); toast.timer=setTimeout(()=>el.classList.add('hidden'),4500); }
