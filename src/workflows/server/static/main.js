document.addEventListener('DOMContentLoaded', () => {
            const workflowSelect = document.getElementById('workflow-select');
            const runButton = document.getElementById('run-button');
            const runsContainer = document.getElementById('runs');
            const eventStreamContainer = document.getElementById('event-stream');

            let activeRunId = null;
            const eventStreams = {};
            let currentSchema = null;

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

                const startEvent = collectFormData()
                const body = {"start_event": JSON.stringify(startEvent), "context": {}, "kwargs": {}}

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
            async function fetchSchema(workflowName) {
                if (!workflowName.trim()) {
                    return null;
                }

                const loadingIndicator = document.getElementById('loading-indicator');
                const errorMessage = document.getElementById('error-message');
                const formFields = document.getElementById('form-fields');

                // Show loading state
                loadingIndicator.classList.remove('d-none');
                errorMessage.classList.add('d-none');
                formFields.innerHTML = '';

                try {
                    const response = await fetch(`/workflows/${encodeURIComponent(workflowName)}/start-event`);

                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }

                    const schema = await response.json();
                    loadingIndicator.classList.add('d-none');

                    return schema.result;
                } catch (error) {
                    loadingIndicator.classList.add('d-none');
                    errorMessage.textContent = `Error fetching schema: ${error.message}`;
                    errorMessage.classList.remove('d-none');
                    console.error('Error fetching schema:', error);
                    return null;
                }
            }

            // Function to generate form fields based on schema
            function generateFormFields(schema) {
                const formFields = document.getElementById('form-fields');
                formFields.innerHTML = '';

                // Check if schema is empty or has no properties
                if (!schema || !schema.properties || Object.keys(schema.properties).length === 0) {
                    // Fall back to original textarea
                    formFields.innerHTML = `
                        <div class="mb-3">
                            <label for="workflow-input" class="form-label">Input (JSON)</label>
                            <textarea id="workflow-input" class="form-control" rows="5" placeholder='{"start_event": "...", "context": {}, "kwargs": {}}'></textarea>
                        </div>
                    `;
                    return;
                }

                // Generate fields based on schema properties
                Object.entries(schema.properties).forEach(([fieldName, fieldSchema]) => {
                    const isRequired = schema.required && schema.required.includes(fieldName);
                    const fieldTitle = fieldSchema.title || fieldName;
                    const fieldType = fieldSchema.type || 'string';
                    const fieldDescription = fieldSchema.description || '';

                    const fieldDiv = document.createElement('div');
                    fieldDiv.className = 'mb-3';

                    let fieldHtml = '';

                    // Create label
                    fieldHtml += `
                        <label for="field-${fieldName}" class="form-label">
                            ${fieldTitle}${isRequired ? ' <span class="text-danger">*</span>' : ''}
                        </label>
                    `;

                    // Create appropriate input based on type
                    if (fieldType === 'string') {
                        fieldHtml += `
                            <textarea
                                id="field-${fieldName}"
                                name="${fieldName}"
                                class="form-control"
                                rows="3"
                                ${isRequired ? 'required' : ''}
                                placeholder="${fieldDescription || `Enter ${fieldTitle.toLowerCase()}`}"
                                onchange="updatePreview()"
                                onkeyup="updatePreview()"
                            ></textarea>
                        `;
                    } else if (fieldType === 'number' || fieldType === 'integer') {
                        fieldHtml += `
                            <input
                                type="number"
                                id="field-${fieldName}"
                                name="${fieldName}"
                                class="form-control"
                                ${isRequired ? 'required' : ''}
                                placeholder="${fieldDescription || `Enter ${fieldTitle.toLowerCase()}`}"
                                ${fieldType === 'integer' ? 'step="1"' : 'step="any"'}
                                onchange="updatePreview()"
                                onkeyup="updatePreview()"
                            >
                        `;
                    } else if (fieldType === 'boolean') {
                        fieldHtml += `
                            <select
                                id="field-${fieldName}"
                                name="${fieldName}"
                                class="form-control"
                                ${isRequired ? 'required' : ''}
                                onchange="updatePreview()"
                            >
                                <option value="">Select...</option>
                                <option value="true">True</option>
                                <option value="false">False</option>
                            </select>
                        `;
                    } else {
                        // Default to textarea for complex types
                        fieldHtml += `
                            <textarea
                                id="field-${fieldName}"
                                name="${fieldName}"
                                class="form-control"
                                rows="3"
                                ${isRequired ? 'required' : ''}
                                placeholder="${fieldDescription || `Enter ${fieldTitle.toLowerCase()} (JSON format)`}"
                                onchange="updatePreview()"
                                onkeyup="updatePreview()"
                            ></textarea>
                        `;
                    }

                    // Add description if available
                    if (fieldDescription) {
                        fieldHtml += `<small class="form-text text-muted">${fieldDescription}</small>`;
                    }

                    fieldDiv.innerHTML = fieldHtml;
                    formFields.appendChild(fieldDiv);
                });

                // Update preview after generating fields
                updatePreview();
            }

            // Function to update JSON preview
            function updatePreview() {
                const preview = document.getElementById('json-preview');
                const formData = collectFormData();
                preview.textContent = JSON.stringify(formData, null, 2);
            }

            document.addEventListener('input', function(e) {
                if (e.target.closest('#form-fields')) {
                    updatePreview();
                }
            });

            // Function to collect form data
            function collectFormData() {
                const formFields = document.querySelectorAll('#form-fields input, #form-fields textarea, #form-fields select');
                const data = {};

                // Check if we're using fallback textarea
                const fallbackTextarea = document.getElementById('workflow-input');
                if (fallbackTextarea) {
                    try {
                        return fallbackTextarea.value ? JSON.parse(fallbackTextarea.value) : {};
                    } catch (e) {
                        return { error: 'Invalid JSON in input field' };
                    }
                }

                // Collect data from dynamic fields
                formFields.forEach(field => {
                    const fieldName = field.name;
                    let value = field.value.trim();

                    if (!value) return;

                    // Try to parse as JSON for complex fields, otherwise use as string
                    if (field.tagName === 'TEXTAREA' && value.startsWith('{') || value.startsWith('[')) {
                        try {
                            data[fieldName] = JSON.parse(value);
                        } catch (e) {
                            data[fieldName] = value;
                        }
                    } else if (field.type === 'number') {
                        data[fieldName] = parseFloat(value);
                    } else if (field.tagName === 'SELECT' && (value === 'true' || value === 'false')) {
                        data[fieldName] = value === 'true';
                    } else {
                        data[fieldName] = value;
                    }
                });

                return data;
            }

            async function handleWorkflowSelectChange() {
                const workflowSelect = document.getElementById('workflow-select');
                const selectedWorkflow = workflowSelect.value;

                if (!selectedWorkflow) {
                    // Reset to fallback form if no workflow selected
                    generateFormFields(null);
                    return;
                }

                const schema = await fetchSchema(selectedWorkflow);
                if (schema) {
                    currentSchema = schema;
                    generateFormFields(schema);
                }
            }

            document.getElementById('workflow-select').addEventListener('change', handleWorkflowSelectChange);
        });
