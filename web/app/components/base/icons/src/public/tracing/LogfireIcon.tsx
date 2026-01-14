import * as React from 'react'

/**
 * Logfire placeholder icon.
 *
 * This intentionally uses a plain white block as a placeholder, because we do not
 * ship an official Logfire logo asset in this repository.
 */
const LogfireIcon = (props: React.SVGProps<SVGSVGElement>) => {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
      {...props}
    >
      <rect x="2" y="2" width="20" height="20" rx="4" fill="#fff" />
    </svg>
  )
}

LogfireIcon.displayName = 'LogfireIcon'

export default LogfireIcon
