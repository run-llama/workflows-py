document.addEventListener('DOMContentLoaded', () => {
    const workflowSelect = document.getElementById('workflow-select');
    const runButton = document.getElementById('run-button');
    const runsContainer = document.getElementById('runs');
    const eventStreamContainer = document.getElementById('event-stream');

    let activeRunId = null;
    const eventStreams = {};
    let currentSchema = null;
    let currentOutput = null;

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

        clearOutputFields()

        const startEvent = collectFormData();
        const body = { "start_event": JSON.stringify(startEvent), "context": {}, "kwargs": {} };

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
        runItem.className = 'p-2 border-b border-gray-200 cursor-pointer hover:bg-gray-100 transition-colors duration-200';
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
                child.classList.remove('hover:bg-gray-100');
            } else {
                child.classList.remove('bg-blue-600', 'text-white');
                child.classList.add('hover:bg-gray-100');
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
                        let eventDetails = "";
                        for (const key in eventData.value) {
                            const value = eventData.value[key];
                            if (!value || value.toString().trim() === '') {
                                eventDetails += `<details class="mb-2"><summary class="cursor-pointer text-gray-700 hover:text-gray-900 font-medium">${key}</summary><p class="mt-1 ml-4 text-gray-600 text-sm">No data</p></details>`;
                            } else {
                                eventDetails += `<details class="mb-2"><summary class="cursor-pointer text-gray-700 hover:text-gray-900 font-medium">${key}</summary><p class="mt-1 ml-4 text-gray-600 text-sm whitespace-pre-wrap break-words">${value}</p></details>`;
                            }
                        }
                        const formattedEvent = `<div class="mb-4 p-3 bg-white rounded border border-gray-200"><strong class="text-gray-800">Event:</strong> <span class="text-blue-600 font-mono text-sm">${eventData.qualified_name}</span><br><strong class="text-gray-800">Data:</strong><div class="mt-2">${eventDetails}</div></div>`;
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

            outputData = await collectOutputData(handlerId)
            populateOutputFields(outputData)
        } catch (err) {
            console.error('Error streaming events:', err);
            eventStreamContainer.innerHTML += `<div class="text-red-600 p-3 bg-red-50 border border-red-200 rounded">Error streaming events: ${err.message}</div>`;
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
        loadingIndicator.classList.remove('hidden');
        errorMessage.classList.add('hidden');
        formFields.innerHTML = '';

        try {
            const response = await fetch(`/workflows/${encodeURIComponent(workflowName)}/schema`);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const schema = await response.json();
            loadingIndicator.classList.add('hidden');

            return schema.start;
        } catch (error) {
            loadingIndicator.classList.add('hidden');
            errorMessage.textContent = `Error fetching schema: ${error.message}`;
            errorMessage.classList.remove('hidden');
            console.error('Error fetching schema:', error);
            return null;
        }
    }

    async function fetchOutputSchema(workflowName) {
        if (!workflowName.trim()) {
            return null;
        }

        const loadingIndicator = document.getElementById('output-loading-indicator');
        const errorMessage = document.getElementById('output-error-message');
        const outFields = document.getElementById('output-fields');

        // Show loading state
        loadingIndicator.classList.remove('hidden');
        errorMessage.classList.add('hidden');
        outFields.innerHTML = '';

        try {
            const response = await fetch(`/workflows/${encodeURIComponent(workflowName)}/schema`);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const schema = await response.json();
            loadingIndicator.classList.add('hidden');

            return schema.stop;
        } catch (error) {
            loadingIndicator.classList.add('hidden');
            errorMessage.textContent = `Error fetching schema: ${error.message}`;
            errorMessage.classList.remove('hidden');
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
                <div class="mb-4">
                    <label for="workflow-input" class="block text-sm font-medium text-gray-700 mb-2">Input (JSON)</label>
                    <textarea
                        id="workflow-input"
                        class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm font-mono"
                        rows="5"
                        placeholder='{"start_event": "...", "context": {}, "kwargs": {}}'
                    ></textarea>
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
            fieldDiv.className = 'mb-4';

            let fieldHtml = '';

            // Create label
            fieldHtml += `
                <label for="field-${fieldName}" class="block text-sm font-medium text-gray-700 mb-2">
                    ${fieldTitle}${isRequired ? ' <span class="text-red-500">*</span>' : ''}
                </label>
            `;

            // Create appropriate input based on type
            if (fieldType === 'string') {
                fieldHtml += `
                    <textarea
                        id="field-${fieldName}"
                        name="${fieldName}"
                        class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                        rows="3"
                        ${isRequired ? 'required' : ''}
                        placeholder="${fieldDescription || `Enter ${fieldTitle.toLowerCase()}`}"
                    ></textarea>
                `;
            } else if (fieldType === 'number' || fieldType === 'integer') {
                fieldHtml += `
                    <input
                        type="number"
                        id="field-${fieldName}"
                        name="${fieldName}"
                        class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                        ${isRequired ? 'required' : ''}
                        placeholder="${fieldDescription || `Enter ${fieldTitle.toLowerCase()}`}"
                        ${fieldType === 'integer' ? 'step="1"' : 'step="any"'}
                    >
                `;
            } else if (fieldType === 'boolean') {
                fieldHtml += `
                    <select
                        id="field-${fieldName}"
                        name="${fieldName}"
                        class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                        ${isRequired ? 'required' : ''}
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
                        class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm font-mono"
                        rows="3"
                        ${isRequired ? 'required' : ''}
                        placeholder="${fieldDescription || `Enter ${fieldTitle.toLowerCase()} (JSON format)`}"
                    ></textarea>
                `;
            }

            // Add description if available
            if (fieldDescription) {
                fieldHtml += `<p class="text-sm text-gray-500 mt-1">${fieldDescription}</p>`;
            }

            fieldDiv.innerHTML = fieldHtml;
            formFields.appendChild(fieldDiv);
        });

        // Update preview after generating fields
        updatePreview();
    }

    // Function to generate form fields based on schema
    function generateOutputFields(schema) {
        const outFields = document.getElementById('output-fields');
        outFields.innerHTML = '';

        // Check if schema is empty or has no properties
        if (!schema || !schema.properties || Object.keys(schema.properties).length === 0) {
            // Fall back to original textarea
            outFields.innerHTML = `
                <div class="mb-4">
                    <label for="workflow-input" class="block text-sm font-medium text-gray-700 mb-2">Output (JSON)</label>
                    <textarea
                        id="workflow-output"
                        class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm font-mono opacity-50 cursor-not-allowed"
                        rows="5"
                        placeholder='{"start_event": "...", "context": {}, "kwargs": {}}'
                    ></textarea>
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
            fieldDiv.className = 'mb-4';

            let fieldHtml = '';

            // Create label
            fieldHtml += `
                <label for="output-${fieldName}" class="block text-sm font-medium text-gray-700 mb-2">
                    ${fieldTitle}
                </label>
            `;

            // Create appropriate input based on type
            if (fieldType === 'string') {
                fieldHtml += `
                    <textarea
                        id="output-${fieldName}"
                        name="${fieldName}"
                        class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm  opacity-50 cursor-not-allowed"
                        rows="3"
                        ${isRequired ? 'required' : ''}
                        placeholder="${fieldDescription || `Enter ${fieldTitle.toLowerCase()}`}"
                    ></textarea>
                `;
            } else if (fieldType === 'number' || fieldType === 'integer') {
                fieldHtml += `
                    <input
                        type="number"
                        id="output-${fieldName}"
                        name="${fieldName}"
                        class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm  opacity-50 cursor-not-allowed"
                        ${isRequired ? 'required' : ''}
                        placeholder="${fieldDescription || `Enter ${fieldTitle.toLowerCase()}`}"
                        ${fieldType === 'integer' ? 'step="1"' : 'step="any"'}
                    >
                `;
            } else if (fieldType === 'boolean') {
                fieldHtml += `
                    <select
                        id="output-${fieldName}"
                        name="${fieldName}"
                        class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm  opacity-50 cursor-not-allowed"
                    >
                        <option value="">Output here...</option>
                        <option value="true">True</option>
                        <option value="false">False</option>
                    </select>
                `;
            } else {
                // Default to textarea for complex types
                fieldHtml += `
                    <textarea
                        id="output-${fieldName}"
                        name="${fieldName}"
                        class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm font-mono opacity-50 cursor-not-allowed"
                        rows="3"
                    ></textarea>
                `;
            }

            // Add description if available
            if (fieldDescription) {
                fieldHtml += `<p class="text-sm text-gray-500 mt-1">${fieldDescription}</p>`;
            }

            fieldDiv.innerHTML = fieldHtml;
            outFields.appendChild(fieldDiv);
        });
    }

    async function collectOutputData(handlerId) {
        try {
            const response = await fetch(`/results/${handlerId}`);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                return { error: `Error while fetching workflow data: [${response.status}] ${JSON.stringify(errorData)}` };
            }

            const workflowResult = await response.json();
            const resultData = workflowResult.result || workflowResult;

            // Populate the output fields with the result data
            populateOutputFields(resultData);

            return resultData;
        } catch (error) {
            console.error('Error collecting output data:', error);
            return { error: `Network error: ${error.message}` };
        }
    }

    function populateOutputFields(data) {
        // Check if we're using the fallback textarea
        const fallbackTextarea = document.getElementById('workflow-output');
        if (fallbackTextarea) {
            fallbackTextarea.value = JSON.stringify(data, null, 2);
            return;
        }

        // Populate dynamic fields based on schema
        const outputFields = document.querySelectorAll('#output-fields input, #output-fields textarea, #output-fields select');

        outputFields.forEach(field => {
            const fieldName = field.name;
            if (data.hasOwnProperty(fieldName)) {
                const value = data[fieldName];

                if (field.tagName === 'TEXTAREA') {
                    // For textareas, handle both simple strings and complex objects
                    if (typeof value === 'object') {
                        field.value = JSON.stringify(value, null, 2);
                    } else {
                        field.value = value || '';
                    }
                } else if (field.type === 'number') {
                    field.value = value !== null && value !== undefined ? value : '';
                } else if (field.tagName === 'SELECT') {
                    // For boolean selects
                    if (typeof value === 'boolean') {
                        field.value = value.toString();
                    } else {
                        field.value = value || '';
                    }
                } else {
                    // For regular inputs
                    field.value = value || '';
                }
            }
        });
    }

    // Function to clear output fields (useful when starting a new workflow)
    function clearOutputFields() {
        const fallbackTextarea = document.getElementById('workflow-output');
        if (fallbackTextarea) {
            fallbackTextarea.value = '';
            return;
        }

        const outputFields = document.querySelectorAll('#output-fields input, #output-fields textarea, #output-fields select');
        outputFields.forEach(field => {
            if (field.tagName === 'SELECT') {
                field.selectedIndex = 0;
            } else {
                field.value = '';
            }
        });
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
            if (field.tagName === 'TEXTAREA' && (value.startsWith('{') || value.startsWith('['))) {
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
            generateOutputFields(null)
            return;
        }

        const startSchema = await fetchSchema(selectedWorkflow);
        const stopSchema = await fetchOutputSchema(selectedWorkflow);
        if (startSchema) {
            currentSchema = startSchema;
            generateFormFields(startSchema);
        }
        if (stopSchema) {
            currentOutput = stopSchema;
            generateOutputFields(stopSchema);
        }
    }

    document.getElementById('workflow-select').addEventListener('change', handleWorkflowSelectChange);

    // Initial form fields
    generateFormFields(null);
});
