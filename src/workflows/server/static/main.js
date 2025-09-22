document.addEventListener('DOMContentLoaded', () => {
    const workflowSelect = document.getElementById('workflow-select');
    const runButton = document.getElementById('run-button');
    const runsContainer = document.getElementById('runs');
    const eventStreamContainer = document.getElementById('event-stream');
    const workflowViz = document.getElementById('workflowViz')
    const nodeDescription = document.getElementById('nodeDescription')

    let activeRunId = null;
    const eventStreams = {};
    const eventSources = {};
    let cy = null;
    let currentSchema = null;
    let currentOutput = null;
    let vizData = null;

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

    nodeDescription.addEventListener('click', (e) => {
        if (e.target && e.target.id === 'hideNodeDescription') {
            nodeDescription.classList.add("hidden");
        }
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
            let finalized = false;
            const eventSource = new EventSource(`/events/${handlerId}?sse=true`);
            eventSources[handlerId] = eventSource;

            const finalize = async () => {
                if (finalized) return;
                finalized = true;
                try {
                    const outputData = await collectOutputData(handlerId);
                    populateOutputFields(outputData);
                } catch (e) {
                    console.error('Error fetching final output:', e);
                } finally {
                    try { eventSource.close(); } catch (_) {}
                }
            };

            eventSource.onmessage = (evt) => {
                try {
                    const eventData = JSON.parse(evt.data);
                    highlightNode(eventData.qualified_name.split(".").at(-1))
                    let eventDetails = "";
                    let currentStepName = "";
                    const formatEventName = (raw) => {
                        if (!raw) return "Not recorded";
                        try {
                            const str = typeof raw === 'string' ? raw : String(raw);
                            const cleaned = str.replace("<class", "").replace(">", "").replaceAll("'", "");
                            const parts = cleaned.split(".");
                            const last = parts.at(-1);
                            return last && last.trim() ? last : "Not recorded";
                        } catch (_) {
                            return "Not recorded";
                        }
                    };
                    if (!eventData.value || typeof eventData.value !== 'object') {
                        // In rare cases, skip rendering details if value is missing
                        console.warn('Unexpected event without value object:', eventData);
                    }
                    for (const key in (eventData.value || {})) {
                        const value = eventData.value[key];
                        if (eventData.qualified_name === "workflows.events.StepStateChanged" && key === "name") {
                            currentStepName = value
                            highlightNode(currentStepName)
                            if (vizData) {
                                for (const element of vizData.elements) {
                                    if (element.data.id === currentStepName) {
                                        element.data.inputEvent = formatEventName(eventData.value["input_event_name"]);
                                        element.data.outputEvent = formatEventName(eventData.value["output_event_name"]);
                                    }
                                }
                            }
                        }
                        if (!value || value.toString().trim() === '') {
                            eventDetails += `<details class="mb-2"><summary class="cursor-pointer text-gray-700 hover:text-gray-900 font-medium">${key}</summary><p class="mt-1 ml-4 text-gray-600 text-sm">No data</p></details>`;
                        } else {
                            eventDetails += `<details class="mb-2"><summary class="cursor-pointer text-gray-700 hover:text-gray-900 font-medium">${key}</summary><p class="mt-1 ml-4 text-gray-600 text-sm whitespace-pre-wrap break-words">${value}</p></details>`;
                        }
                    }
                    const formattedEvent = `<div class="mb-4 p-3 bg-white rounded border border-gray-200"><strong class="text-gray-800">Event:</strong> <span class="text-blue-600 font-mono text-sm">${eventData.qualified_name}</span><br><strong class="text-gray-800">Data:</strong><div class="mt-2">${eventDetails}</div></div>`;
                    eventStreams[handlerId].push(formattedEvent);
                    requestAnimationFrame(() => {
                        setTimeout(() => {
                            resetNode(eventData.qualified_name.replace("__main__.", "").replace("workflows.events.", ""))
                            resetNode(currentStepName)
                        }, 500); // Still add a small delay for visibility
                    });

                    if (handlerId === activeRunId) {
                        eventStreamContainer.innerHTML += formattedEvent;
                        eventStreamContainer.scrollTop = eventStreamContainer.scrollHeight;
                    }
                } catch (e) {
                    console.error('Error parsing SSE event:', evt.data, e);
                }
            };

            eventSource.onerror = async (err) => {
                // When the stream closes normally, browsers fire onerror and readyState becomes CLOSED
                if (eventSource.readyState === EventSource.CLOSED) {
                    await finalize();
                } else {
                    console.info("Event source disconnected. Ready state:", eventSource.readyState, "error:", err);
                }
            };
        } catch (err) {
            console.error('Error initializing EventSource:', err);
            eventStreamContainer.innerHTML += `<div class="text-red-600 p-3 bg-red-50 border border-red-200 rounded">Error initializing event stream: ${err.message}</div>`;
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
            generateOutputFields(null);
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

        fetchWorkflowViz(selectedWorkflow)
    }

    async function fetchWorkflowViz(selectedWorkflow) {
        const workflowViz = document.getElementById('workflowViz');

        try {
            const response = await fetch(`/workflows/${selectedWorkflow}/representation`);

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                workflowViz.innerHTML = `<p class="text-red-500 text-lg">An error occurred while retrieving the visualization for the workflow<br>${response.status}: ${JSON.stringify(errData)}<br>Try with another workflow</p>`;
                workflowViz.classList.replace("bg-gray-50", "bg-red-300");
                return;
            }

            const vizJson = await response.json();
            const vizDataRaw = vizJson.graph;
            const vizElements = []
            for (const node of vizDataRaw.nodes) {
                node_data = {
                    "data": {
                        "id": node.id,
                        "label": node.label,
                        "type": node.node_type,
                    },
                }

                if (node.node_type === "step") {
                    node_data.classes = "node-step"
                } else if (node.node_type === "event") {
                    node_data.classes = "node-event"
                } else if (node.node_type === "external") {
                    node_data.classes = "node-external"
                } else {
                    node_data.classes = "node"
                }

                if (node.title) {
                    node_data.data.title = node.title
                }

                if (node.event_type) {
                    node_data.data.event_type = node.event_type
                }
                vizElements.push(node_data)
            }
            for (const edge of vizDataRaw.edges) {
                edge_data = {
                    "data": {
                        "id": `${edge.source}-${edge.target}`,
                        "source": edge.source,
                        "target": edge.target,
                    }
                }
                vizElements.push(edge_data)
            }

            vizData = {"elements": vizElements}

            // Clear the container and reset styling
            workflowViz.innerHTML = '';
            workflowViz.classList.replace("bg-red-300", "bg-gray-50");

            // Set container height for Cytoscape
            workflowViz.style.height = '600px';
            workflowViz.style.width = '100%';

            // Initialize Cytoscape
            cy = cytoscape({
                container: workflowViz,

                elements: vizData.elements,

                style: [
                    // Default styles if not provided by backend
                    {
                        selector: '.node-step',
                        style: {
                            'background-color': '#92AEFF',
                            'label': 'data(label)',
                            'text-valign': 'top',
                            'text-halign': 'center',
                            'color': 'white',
                            'text-outline-width': 1,
                            'text-outline-color': '#3E18F9',
                            'shape': 'rectangle',
                            'width': 'label',
                            'height': 40,
                            'font-size': '14px',
                            'font-weight': 'bold',
                            'padding': '10px',
                            'border-width': 1,
                            'border-color': '#2980b9',
                            'border-style': 'solid'
                        }
                    },
                    {
                        selector: 'node',
                        style: {
                            'label': 'data(label)',
                            'text-valign': 'top',
                            'text-halign': 'center',
                            'color': 'white',
                            'font-size': '12px',
                            'font-weight': 'bold',
                            'text-outline-width': 1,
                            'text-outline-color': '#000'
                        }
                    },
                    {
                        selector: '.node-event',
                        style: {
                            'background-color': '#FDEDBA',
                            'label': 'data(label)',
                            'text-valign': 'top',
                            'text-halign': 'center',
                            'color': 'white',
                            'text-outline-width': 1,
                            'text-outline-color': '#FF8705',
                            'shape': 'diamond',
                            'width': 'label',
                            'height': 'label',
                            'font-size': '12px',
                            'font-weight': 'bold',
                            'padding': '10px',
                            'border-width': 1,
                            'border-color': '#2980b9',
                            'border-style': 'solid'
                        }
                    },
                    {
                        selector: '.node-external',
                        style: {
                            'background-color': '#f39c12',
                            'label': 'data(label)',
                            'text-valign': 'top',
                            'text-halign': 'center',
                            'color': 'white',
                            'shape': 'diamond',
                            'width': 'label',
                            'height': 'label',
                            'font-size': '12px',
                            'font-weight': 'bold',
                            'padding': '15px'
                        }
                    },
                    {
                        selector: '.workflow-edge',
                        style: {
                            'curve-style': 'bezier',
                            'target-arrow-shape': 'triangle',
                            'arrow-scale': 1.5,
                            'line-color': '#666',
                            'target-arrow-color': '#666',
                            'width': 2
                        }
                    },
                    {
                        selector: 'edge',
                        style: {
                            'curve-style': 'bezier',
                            'target-arrow-shape': 'triangle',
                            'arrow-scale': 1.5,
                            'line-color': '#666',
                            'target-arrow-color': '#666',
                            'width': 2
                        }
                    },
                    {
                        selector: '.node-step:selected',
                        style: {
                            'background-color': '#4B72FE',
                            'border-width': 3,
                            'border-style': 'solid',
                            'border-opacity': 0.8
                        }
                    },
                    {
                        selector: '.node-event:selected',
                        style: {
                            'background-color': '#FFBD74',
                            'border-width': 3,
                            'border-style': 'solid',
                            'border-opacity': 0.8
                        }
                    },
                    {
                        selector: 'node:selected',
                        style: {
                            'border-width': 3,
                            'border-color': '#34495e',
                            'border-opacity': 0.8
                        }
                    }
                ],

                layout: {
                    name: 'dagre',
                    directed: true,
                    rankDir: 'LR', // Left to Right
                    spacingFactor: 1.5,
                    nodeSep: 50,
                    rankSep: 100,
                    padding: 30
                },

                // Interactive options
                userZoomingEnabled: true,
                userPanningEnabled: true,
                boxSelectionEnabled: false,
                selectionType: 'single',
                touchTapThreshold: 8,
                desktopTapThreshold: 4,
                autolock: false,
                autoungrabify: false,
                autounselectify: false
            });

            // Add click event for nodes (optional - shows node info)
            cy.on('tap', 'node', function(event) {
                const node = event.target;
                const data = node.data();

                let info = `<strong class="text-center text-xl">Node ${data.label}</strong><br><strong>Type</strong>: ${data.type}`;
                if (data.title) info += `<br><strong>Title</strong>: ${data.title}`;
                if (data.event_type) info += `<br><strong>Event Type</strong>: ${data.event_type}`;
                if (data.inputEvent) info += `<br><strong>Input Event (last call)</strong>: ${data.inputEvent}`
                if (data.outputEvent) info += `<br><strong>Output Event (last call)</strong>: ${data.outputEvent}<br>`

                // Remove all possible background colors first
                nodeDescription.classList.remove("bg-gray-50", "bg-[#92AEFF]", "bg-[#FDEDBA]", "bg-[#FFBFF8]");
                nodeDescription.classList.remove("hidden")

                let bgButton = "bg-blue-600"
                let txtButton = "text-white"
                // Then add the correct one
                if (data.type === "step") {
                    nodeDescription.classList.add("bg-[#92AEFF]");
                    bgButton = "bg-[#4B72FE]"
                } else if (data.type === "event") {
                    nodeDescription.classList.add("bg-[#FDEDBA]");
                    bgButton = "bg-[#FFBD74]"
                    txtButton = "text-gray-700"
                } else {
                    nodeDescription.classList.add("bg-[#FFBFF8]");
                }

                nodeDescription.innerHTML = `<p class="text-gray-700 text-lg p-4">${info}</p><br><button id="hideNodeDescription" class="w-full ${bgButton} ${txtButton} font-bold text-center rounded-lg shadow-lg">Hide Description</button>`;
            });

            // Fit the graph to the container
            cy.fit();
            cy.center();

        } catch (error) {
            console.error('Error fetching workflow visualization:', error);
            workflowViz.innerHTML = `<p class="text-red-500 text-lg">An unexpected error occurred while loading the workflow visualization.<br>Please try again.</p>`;
            workflowViz.classList.replace("bg-gray-50", "bg-red-300");
        }
    }

    // Change a specific node by ID
    function highlightNode(nodeId, color = '#43BF69') {
        const node = cy.getElementById(nodeId);
        if (node.length === 0) {
            return;
        }
        node.style({
            'background-color': color,
            'border-width': 3,
            'text-outline-color': color,
        });
    }

    // Reset node to original style
    function resetNode(nodeId) {
        const node = cy.getElementById(nodeId);
        if (node.length === 0) {
            return;
        }
        node.removeStyle();
    }


    document.getElementById('workflow-select').addEventListener('change', handleWorkflowSelectChange);

    // Initial form fields
    generateFormFields(null);
});
