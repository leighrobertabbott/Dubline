(() => {
  const ui = { info: null, library: null, step: 'check', timer: null, forced: false, choicesTouched: false };
  const find = id => document.getElementById(id);

  document.addEventListener('DOMContentLoaded', () => {
    find('setupButton').addEventListener('click', () => openSetup(true));
    find('setupClose').addEventListener('click', closeSetup);
    find('setupToAccess').addEventListener('click', () => showStep('access'));
    find('setupRescan').addEventListener('click', rescan);
    find('setupBackToCheck').addEventListener('click', () => showStep('check'));
    find('showToken').addEventListener('click', toggleToken);
    find('saveToken').addEventListener('click', saveToken);
    find('hfForget').addEventListener('click', forgetToken);
    find('showTmdbToken').addEventListener('click', () => toggleSecret('tmdbToken', 'showTmdbToken'));
    find('saveTmdbToken').addEventListener('click', saveTmdbToken);
    find('tmdbForget').addEventListener('click', forgetTmdbToken);
    find('enhancedSpeakers').addEventListener('change', () => { ui.choicesTouched = true; });
    find('selectiveLipSync').addEventListener('change', () => { ui.choicesTouched = true; });
    find('automaticArtwork').addEventListener('change', () => { ui.choicesTouched = true; });
    find('beginSetup').addEventListener('click', beginInstall);
    find('retrySetup').addEventListener('click', beginInstall);
    find('cancelSetup').addEventListener('click', cancelInstall);
    find('finishSetup').addEventListener('click', closeSetup);
    document.addEventListener('keydown', handleDialogKeys);
    openSetup(false);
  });

  async function setupApi(url, options) {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Setup could not complete that request.');
    return data;
  }

  async function openSetup(forced) {
    ui.forced = forced;
    try {
      [ui.info, ui.library] = await Promise.all([setupApi('/api/setup'), setupApi('/api/library/settings')]);
      renderSetup();
      if (forced || ui.info.first_run || ui.info.running || ui.info.phase === 'error') {
        find('setupWizard').classList.remove('hidden');
        document.body.style.overflow = 'hidden';
        document.querySelector('.app-shell').inert = true;
        showStep(ui.info.running || ['error', 'complete'].includes(ui.info.phase) ? 'install' : 'check');
        setTimeout(() => {
          const target = !find('setupRescan').classList.contains('hidden') ? find('setupRescan')
            : !find('setupToAccess').disabled ? find('setupToAccess') : find('setupClose');
          target.focus();
        }, 50);
      }
      scheduleRefresh();
    } catch (error) {
      const system = find('systemState')?.querySelector('span');
      if (system) system.textContent = 'Setup service offline';
    }
  }

  function closeSetup() {
    if (!ui.info) return;
    if (ui.info.first_run && ui.info.platform?.supported && !ui.info.ready) return;
    find('setupWizard').classList.add('hidden');
    document.body.style.overflow = '';
    document.querySelector('.app-shell').inert = false;
    ui.forced = false;
    find('setupButton').focus();
    if (typeof loadSystem === 'function') loadSystem();
  }

  function showStep(step) {
    ui.step = step;
    find('setupAccessPanel').closest('.setup-main').scrollTop = 0;
    for (const name of ['check', 'access', 'install']) {
      find(`setup${name[0].toUpperCase()}${name.slice(1)}Panel`).classList.toggle('hidden', name !== step);
      const marker = document.querySelector(`[data-setup-step="${name}"]`);
      const order = ['check', 'access', 'install'];
      marker.classList.toggle('active', name === step);
      marker.classList.toggle('done', order.indexOf(name) < order.indexOf(step));
    }
    if (step === 'access') setTimeout(() => find('setupAccessTitle').focus(), 50);
  }

  function renderSetup() {
    const info = ui.info;
    if (!info) return;
    const system = info.system;
    const platform = info.platform;
    const cards = find('setupSystemGrid').children;
    setSystemCard(cards[0], platform.supported, `${platform.os} · ${platform.architecture}`, platform.supported ? 'Supported' : 'Not supported by this release');
    setSystemCard(cards[1], system.cuda, system.gpu, system.cuda ? 'CUDA available' : 'NVIDIA CUDA required');
    setSystemCard(cards[2], system.enough_disk, `${system.disk_free_gb} GB free`, info.missing_download_gb ? `${info.missing_download_gb} GB still needed` : 'Downloads present');
    const compatibility = find('setupCompatibility');
    const incompatible = !platform.supported || !system.cuda || !system.enough_disk;
    compatibility.classList.toggle('hidden', !incompatible);
    compatibility.querySelector('p').textContent = !platform.supported || !system.cuda
      ? platform.message
      : `Free at least ${(info.missing_download_gb + 3).toFixed(1)} GB on this drive, then scan again.`;
    find('setupDownloadEstimate').textContent = info.missing_download_gb
      ? `${info.missing_download_gb} GB to download · resumable`
      : 'Everything required is already installed';
    find('setupToAccess').disabled = !info.can_install && !info.ready;
    find('setupRescan').classList.toggle('hidden', !incompatible || !platform.supported);
    find('setupToAccess').textContent = info.ready ? 'Manage optional models' : 'Continue';
    find('setupClose').classList.toggle('hidden', !info.ready && platform.supported);

    const connected = info.token.configured;
    find('hfConnectionState').textContent = connected ? `Connected · ${info.token.display}` : 'Not connected';
    find('hfInstructions').classList.toggle('hidden', connected);
    find('hfForget').classList.toggle('hidden', !connected);
    if (!ui.choicesTouched) find('enhancedSpeakers').checked = connected;
    find('enhancedSpeakers').disabled = !connected;
    const catalogueConnected = Boolean(ui.library?.configured);
    find('tmdbConnectionState').textContent = catalogueConnected ? `${ui.library.managed ? 'Included' : 'Developer override'} · ${ui.library.display}` : 'TV lookup only in this checkout';
    find('tmdbInstructions').classList.toggle('hidden', catalogueConnected);
    find('tmdbForget').classList.toggle('hidden', !catalogueConnected || Boolean(ui.library?.managed));
    if (!ui.choicesTouched) find('automaticArtwork').checked = ui.library?.enabled !== false;
    renderComponents();
    renderInstallState();
  }

  function setSystemCard(card, good, title, note) {
    card.classList.remove('checking', 'good', 'bad');
    card.classList.add(good ? 'good' : 'bad');
    card.querySelector('strong').textContent = title;
    card.querySelector('small').textContent = note;
  }

  function renderComponents() {
    const list = find('setupComponents');
    list.replaceChildren();
    for (const component of ui.info.components) {
      const item = document.createElement('div');
      item.className = `setup-component${component.ready ? ' ready' : ''}${ui.info.active_component === component.key ? ' active' : ''}${component.required ? '' : ' optional'}`;
      const state = component.ready ? '✓' : (ui.info.active_component === component.key ? '' : '·');
      item.innerHTML = `<i>${state}</i><div><strong></strong><small></small></div>`;
      item.querySelector('strong').textContent = component.name;
      item.querySelector('small').textContent = component.ready ? 'Ready' : (component.required ? `${component.estimated_gb} GB` : `Optional · ${component.estimated_gb} GB`);
      list.appendChild(item);
    }
  }

  function renderInstallState() {
    const info = ui.info;
    const complete = info.ready && !info.running && info.phase !== 'error';
    const percent = complete ? 100 : Number(info.progress || 0);
    find('setupProgressBar').style.width = `${percent}%`;
    find('setupProgress').setAttribute('aria-valuenow', String(Math.round(percent)));
    find('setupProgressPercent').textContent = `${Math.round(percent)}%`;
    find('setupProgressLabel').textContent = complete ? 'All components verified' : (info.detail || 'Preparing');
    find('setupInstallDetail').textContent = complete
      ? 'Every required component passed its check.'
      : info.running ? 'Downloads resume where they stopped if the connection drops.'
      : info.phase === 'error' ? 'Completed downloads were kept. Fix the note below to continue from the same point.'
      : 'Ready to download.';
    find('setupInstallKicker').textContent = complete ? 'DONE' : info.phase === 'error' ? 'PAUSED' : 'STEP 3 OF 3';
    find('setupInstallTitle').textContent = complete ? 'Ready to dub.' : info.phase === 'error' ? 'One step needs attention' : 'Installing';
    const error = find('setupError');
    error.classList.toggle('hidden', info.phase !== 'error');
    error.querySelector('p').textContent = info.error || '';
    const log = find('setupLog');
    log.classList.toggle('hidden', !info.logs?.length);
    log.querySelector('div').textContent = (info.logs || []).join('\n');
    find('cancelSetup').classList.toggle('hidden', !info.running);
    find('retrySetup').classList.toggle('hidden', info.phase !== 'error');
    find('finishSetup').classList.toggle('hidden', !complete);
    find('setupKeepOpen').classList.toggle('hidden', complete || info.phase === 'error');
  }

  function toggleToken() {
    toggleSecret('hfToken', 'showToken');
  }

  function toggleSecret(inputId, buttonId) {
    const input = find(inputId);
    input.type = input.type === 'password' ? 'text' : 'password';
    find(buttonId).textContent = input.type === 'password' ? 'Show' : 'Hide';
  }

  async function saveToken() {
    const button = find('saveToken');
    const token = find('hfToken').value.trim();
    if (!token) return showSetupMessage('Paste a Hugging Face token first.');
    button.disabled = true;
    button.textContent = 'Checking…';
    try {
      await setupApi('/api/setup/token', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({token})});
      find('hfToken').value = '';
      ui.info = await setupApi('/api/setup');
      ui.choicesTouched = false;
      renderSetup();
    } catch (error) { showSetupMessage(error.message); }
    finally { button.disabled = false; button.textContent = 'Connect'; }
  }

  async function forgetToken() {
    try {
      await setupApi('/api/setup/token', {method:'DELETE'});
      ui.info = await setupApi('/api/setup');
      ui.choicesTouched = false;
      renderSetup();
    } catch (error) { showSetupMessage(error.message); }
  }

  async function saveTmdbToken() {
    const button = find('saveTmdbToken');
    const token = find('tmdbToken').value.trim();
    if (!token) return showSetupMessage('Paste a TMDB API Read Access Token first.');
    button.disabled = true;
    button.textContent = 'Checking…';
    try {
      ui.library = await setupApi('/api/library/token', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({token})});
      find('tmdbToken').value = '';
      renderSetup();
      if (typeof loadJobs === 'function') loadJobs();
    } catch (error) { showSetupMessage(error.message); }
    finally { button.disabled = false; button.textContent = 'Save override'; }
  }

  async function forgetTmdbToken() {
    try {
      ui.library = await setupApi('/api/library/token', {method:'DELETE'});
      renderSetup();
    } catch (error) { showSetupMessage(error.message); }
  }

  async function rescan() {
    const button = find('setupRescan');
    button.disabled = true;
    button.textContent = 'Scanning…';
    try {
      [ui.info, ui.library] = await Promise.all([setupApi('/api/setup'), setupApi('/api/library/settings')]);
      renderSetup();
      if (typeof loadSystem === 'function') loadSystem();
    } catch (error) { showSetupMessage(error.message); }
    finally { button.disabled = false; button.textContent = 'Scan again'; }
  }

  async function beginInstall() {
    const enhanced = find('enhancedSpeakers').checked;
    if (enhanced && !ui.info.token.configured) return showSetupMessage('Connect Hugging Face above, or switch enhanced speaker detection off.');
    const button = find('beginSetup');
    button.disabled = true;
    try {
      ui.info = await setupApi('/api/setup/install', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({enhanced_speakers:enhanced, selective_lip_sync:find('selectiveLipSync').checked, media_lookup:find('automaticArtwork').checked})});
      showStep('install');
      renderSetup();
      scheduleRefresh();
    } catch (error) { showSetupMessage(error.message); }
    finally { button.disabled = false; }
  }

  async function cancelInstall() {
    try {
      ui.info = await setupApi('/api/setup/cancel', {method:'POST'});
      renderSetup();
      scheduleRefresh();
    } catch (error) { showSetupMessage(error.message); }
  }

  function showSetupMessage(message) {
    if (typeof toast === 'function') toast(message);
    else window.alert(message);
  }

  function handleDialogKeys(event) {
    const wizard = find('setupWizard');
    if (wizard.classList.contains('hidden')) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      closeSetup();
      return;
    }
    if (event.key !== 'Tab') return;
    const focusable = [...wizard.querySelectorAll('a[href],button:not([disabled]),input:not([disabled]),summary,[tabindex]:not([tabindex="-1"])')]
      .filter(element => !element.closest('.hidden') && element.getClientRects().length);
    if (!focusable.length) return;
    const first = focusable[0], last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  }

  function scheduleRefresh() {
    clearTimeout(ui.timer);
    if (!ui.info?.running) return;
    ui.timer = setTimeout(async () => {
      try {
        ui.info = await setupApi('/api/setup');
        renderSetup();
        if (ui.info.running || ui.info.phase === 'error' || ui.info.phase === 'complete') showStep('install');
      } finally { scheduleRefresh(); }
    }, 1100);
  }
})();
