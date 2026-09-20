import React from 'react';

const TEAM_MEMBERS = [
  {
    initials: 'PO',
    index: '01',
    name: 'pontro',
    url: 'https://github.com/pontro',
  },
  {
    initials: 'CH',
    index: '02',
    name: 'Chewbaccas',
    url: 'https://github.com/Chewbaccas',
  },
  {
    initials: 'KA',
    index: '03',
    name: 'katyazano',
    url: 'https://github.com/katyazano',
  },
  {
    initials: 'CR',
    index: '04',
    name: 'cris-hernandezz',
    url: 'https://github.com/cris-hernandezz',
  },
];

export const TeamRail: React.FC = () => {
  return (
    <>
      {/* Rotated Vertical Team Members Ribbon Grid Cell */}
      <div className="team-vertical-label">
        <div className="team-vertical-text">
          <span className="dot" />
          <span>TEAM MEMBERS</span>
        </div>
      </div>

      {/* Right Rail with 4 Monogram Grid Cells */}
      <aside className="right-rail">
        {TEAM_MEMBERS.map((member) => (
          <a
            key={member.initials}
            href={member.url}
            target="_blank"
            rel="noopener noreferrer"
            className="rail-cell"
            title={`${member.name} (GitHub)`}
          >
            <div className="rail-cell-expand">
              <span>{member.name}</span>
              <i className="fa-solid fa-arrow-up-right-from-square text-[9px] text-[#7B8290]"></i>
            </div>
            <div className="rail-cell-icon">
              <span className="rail-cell-initials">{member.initials}</span>
              <span className="rail-cell-index">{member.index}</span>
            </div>
          </a>
        ))}
      </aside>
    </>
  );
};
