const ProtectedTabs = ({ children }) => {
  return (
    <div className="tabs-container">
      <style jsx>{`
        .tabs-container {
          margin: 1rem 0;
        }

        .tabs-radio {
          display: none;
        }

        .tabs-labels {
          display: flex;
          gap: 0.25rem;
          border-bottom: 1px solid #374151;
          margin-bottom: 1rem;
        }

        .tab-label {
          padding: 0.5rem 0.5rem;
          font-weight: 500;
          font-size: 1rem;
          position: relative;
          background: none;
          border: none;
          cursor: pointer;
          color: #9ca3af;
          transition: color 0.2s;
          text-align: center;
          min-width: 0;
          flex: 1;
          white-space: normal;
          word-break: break-word;
        }

        .tab-label:hover {
          color: #d1d5db;
        }

        .tabs-radio:checked + .tab-label {
          color: #1f2937;
          font-weight: 600;
        }

        html[data-theme='dark'] .tabs-radio:checked + .tab-label {
          color: white;
        }

        .tabs-radio:checked + .tab-label::after {
          content: '';
          position: absolute;
          bottom: 0;
          left: 0;
          right: 0;
          height: 2px;
          background-color: #a855f7;
        }

        .tabs-content {
          padding: 0.5rem 0;
        }

        .tab-item {
          display: none;
        }

        .tabs-labels:has(#tab-0:checked) ~ .tabs-content .tab-item:nth-child(1),
        .tabs-labels:has(#tab-1:checked) ~ .tabs-content .tab-item:nth-child(2),
        .tabs-labels:has(#tab-2:checked) ~ .tabs-content .tab-item:nth-child(3),
        .tabs-labels:has(#tab-3:checked) ~ .tabs-content .tab-item:nth-child(4) {
          display: block;
        }
      `}</style>

      <div className="tabs-labels">
        <input type="radio" id="tab-0" name="tabs" className="tabs-radio" defaultChecked />
        <label htmlFor="tab-0" className="tab-label">OpenAI w/ OIDC Auth</label>

        <input type="radio" id="tab-1" name="tabs" className="tabs-radio" />
        <label htmlFor="tab-1" className="tab-label">Azure OpenAI w/ OIDC Auth</label>

        <input type="radio" id="tab-2" name="tabs" className="tabs-radio" />
        <label htmlFor="tab-2" className="tab-label">OpenAI w/ Basic Auth</label>

        <input type="radio" id="tab-3" name="tabs" className="tabs-radio" />
        <label htmlFor="tab-3" className="tab-label">Azure OpenAI w/ Basic Auth</label>
      </div>

      <div className="tabs-content">
        {children}
      </div>
    </div>
  );
};

const ProtectedTabItem = ({ label, value, children }) => {
  return <div className="tab-item">{children}</div>;
};

export { ProtectedTabs, ProtectedTabItem };
