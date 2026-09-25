import { LeftOutlined } from '@ant-design/icons'

// React counterpart of power-trade/business/PageShell's separate back header.
export default function DetailHeader({ onBack, backLabel = '返回列表', title, subtitle, tags, actions }) {
  return <header className="detail-header">
    <div className="detail-back-header">
      <button type="button" onClick={onBack}><LeftOutlined aria-hidden="true" /><span>{backLabel}</span></button>
    </div>
    <div className="detail-heading-row">
      <div className="detail-heading">
        <div className="detail-title-row"><h1>{title}</h1>{tags}</div>
        {subtitle && <div className="detail-subtitle">{subtitle}</div>}
      </div>
      {actions && <div className="detail-actions">{actions}</div>}
    </div>
  </header>
}
