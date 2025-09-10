document.addEventListener('DOMContentLoaded', () => {
            const workflowSelect = document.getElementById('workflow-select');
            const runButton = document.getElementById('run-button');
            const workflowInput = document.getElementById('workflow-input');
            const runsContainer = document.getElementById('runs');
            const eventStreamContainer = document.getElementById('event-stream');

            let activeRunId = null;
            const eventStreams = {};

            // Fetch workflows on page load
            fetch('/workflows')
                .then(response => response.json())
                .then(data => {
                    if (data.workflows && data.workflows.length > 0) {
                        data.workflows.forEach(name => {
                            const option = document.createElement('option');
                            option.value = name;
                            option.textContent = name;
                            workflowSelect.appendChild(option);
                        });
                    } else {
                        const option = document.createElement('option');
                        option.textContent = "No workflows found";
                        option.disabled = true;
                        workflowSelect.appendChild(option);
                        runButton.disabled = true;
                    }
                })
                .catch(err => {
                    console.error("Error fetching workflows:", err);
                    const option = document.createElement('option');
                    option.textContent = "Error loading workflows";
                    option.disabled = true;
                    workflowSelect.appendChild(option);
                    runButton.disabled = true;
                });

            runButton.addEventListener('click', () => {
                const workflowName = workflowSelect.value;
                if (!workflowName) {
                    alert('Please select a workflow.');
                    return;
                }

                let body = {};
                try {
                    if (workflowInput.value.trim()) {
                        body = JSON.parse(workflowInput.value);
                    }
                } catch (e) {
                    alert('Invalid JSON input.');
                    return;
                }

                fetch(`/workflows/${workflowName}/run-nowait`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(body),
                })
                .then(response => response.json())
                .then(data => {
                    if (data.handler_id) {
                        addRun(data.handler_id, workflowName);
                        streamEvents(data.handler_id);
                    } else {
                        alert('Failed to start workflow.');
                    }
                })
                .catch(error => {
                    console.error('Error starting workflow:', error);
                    alert('Error starting workflow.');
                });
            });

            function addRun(handlerId, workflowName) {
                const runItem = document.createElement('div');
                runItem.className = 'run-item';
                runItem.textContent = `${workflowName} - ${handlerId}`;
                runItem.dataset.handlerId = handlerId;

                runItem.addEventListener('click', () => {
                    setActiveRun(handlerId);
                });

                runsContainer.prepend(runItem);
                setActiveRun(handlerId);
            }

            function setActiveRun(handlerId) {
                activeRunId = handlerId;

                Array.from(runsContainer.children).forEach(child => {
                    if (child.dataset.handlerId === handlerId) {
                        child.classList.add('active');
                    } else {
                        child.classList.remove('active');
                    }
                });

                eventStreamContainer.innerHTML = '';
                if (eventStreams[handlerId]) {
                    eventStreams[handlerId].forEach(event => {
                        eventStreamContainer.innerHTML += event + '<hr>';
                    });
                    eventStreamContainer.scrollTop = eventStreamContainer.scrollHeight;
                }
            }

            async function streamEvents(handlerId) {
                eventStreams[handlerId] = [];
                try {
                    const response = await fetch(`/events/${handlerId}`);
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let buffer = '';

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) {
                            break;
                        }
                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\n');
                        buffer = lines.pop(); // Keep incomplete line in buffer

                        for (const line of lines) {
                            if (line.trim() === '') continue;
                            try {
                                const eventData = JSON.parse(line);
                                var eventDetails = "";
                                for (const key in eventData.value) {
                                    const value = eventData.value[key];
                                    if (!value || value.toString().trim() === '') {
                                        eventDetails += `<details><summary class="detailsSummary">${key}</summary><p class="detailsP">No data</p></details>`;
                                    } else {
                                        eventDetails += `<details class="details"><summary class="detailsSummary">${key}</summary><p class="detailsP">${value}</p></details>`;
                                    }
                                }
                                const formattedEvent = `<strong>Event:</strong> ${eventData.qualified_name}<br><strong>Data:</strong> ${eventDetails}`;
                                eventStreams[handlerId].push(formattedEvent);

                                if (handlerId === activeRunId) {
                                    eventStreamContainer.innerHTML += formattedEvent + '<hr>';
                                    eventStreamContainer.scrollTop = eventStreamContainer.scrollHeight;
                                }
                            } catch (e) {
                                console.error('Error parsing event line:', line, e);
                            }
                        }
                    }
                } catch (error) {
                    console.error('Streaming failed:', error);
                    const errorMsg = 'Event stream closed or failed.';
                    eventStreams[handlerId].push(errorMsg);
                    if (handlerId === activeRunId) {
                        eventStreamContainer.innerHTML += errorMsg + '<hr>';
                    }
                }
            }
        });
