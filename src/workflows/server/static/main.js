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
        runItem.className = 'p-2 border-b border-gray-200 cursor-pointer hover:bg-gray-100';
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
                child.classList.add('bg-blue-600', 'text-white');
            } else {
                child.classList.remove('bg-blue-600', 'text-white');
            }
        });

        eventStreamContainer.innerHTML = '';
        if (eventStreams[handlerId]) {
            eventStreams[handlerId].forEach(eventHTML => {
                eventStreamContainer.innerHTML += eventHTML;
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
                        const formattedEvent = `
                            <div class="p-2 border-b border-gray-200">
                                <div><strong>Event:</strong> ${eventData.qualified_name}</div>
                                <pre class="text-xs bg-gray-100 p-2 rounded mt-1"><code>${JSON.stringify(eventData.value, null, 2)}</code></pre>
                            </div>`;
                        eventStreams[handlerId].push(formattedEvent);

                        if (handlerId === activeRunId) {
                            eventStreamContainer.innerHTML += formattedEvent;
                            eventStreamContainer.scrollTop = eventStreamContainer.scrollHeight;
                        }
                    } catch (e) {
                        console.error('Error parsing event line:', line, e);
                    }
                }
            }
        } catch (error) {
            console.error('Streaming failed:', error);
            const errorMsg = '<div class="p-2 text-red-500">Event stream closed or failed.</div>';
            eventStreams[handlerId].push(errorMsg);
            if (handlerId === activeRunId) {
                eventStreamContainer.innerHTML += errorMsg;
            }
        }
    }
});
